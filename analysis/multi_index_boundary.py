#!/usr/bin/env python3
"""
Multi-Index Boundary Detection using K-means clustering.

Projects 64D embeddings onto multiple vegetation indices (NDVI, SAVI, BSI)
and uses K-means clustering in the 3D projected space for boundary detection.

This approach preserves more discriminative information than single-index
projection by learning 3 independent directions in embedding space.

Algorithm:
    1. Load embeddings (64D Google Satellite Embeddings) - downsample to 30m
    2. Compute NDVI, SAVI, BSI from Landsat imagery
    3. Fit MultiOutput Ridge regression: embeddings -> [NDVI, SAVI, BSI]
    4. Project embeddings to 3D index space
    5. Apply K-means clustering (k=2) in 3D space
    6. Identify vegetation cluster (higher mean projected values)
    7. Export boundary

Usage:
    python analysis/multi_index_boundary.py \
        --embeddings data/google_embedding_2022.tif \
        --landsat data/LC08_20221001_SAVI.tif
"""

import os
import sys
import argparse
from datetime import datetime
import time

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from skimage import filters

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from analysis.feature_extractors.vegetation_indices import (
    compute_ndvi, compute_savi, compute_bsi
)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Target indices for multi-index projection
TARGET_INDICES = ['NDVI', 'SAVI', 'BSI']

# Landsat 8/9 band indices (1-indexed in rasterio)
LANDSAT_BLUE_BAND = 2   # Band 2: Blue (0.45-0.51 μm)
LANDSAT_RED_BAND = 4    # Band 4: Red (0.64-0.67 μm)
LANDSAT_NIR_BAND = 5    # Band 5: NIR (0.85-0.88 μm)
LANDSAT_SWIR1_BAND = 6  # Band 6: SWIR1 (1.57-1.65 μm)


# =============================================================================
# Progress Printing Utilities
# =============================================================================

def print_step(step_num: int, total_steps: int, description: str):
    """Print a major step header."""
    print(f"\n[{step_num}/{total_steps}] {description}", flush=True)
    print("-" * 50, flush=True)


def print_progress(message: str, indent: int = 1):
    """Print a progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "  " * indent
    print(f"{prefix}[{timestamp}] {message}", flush=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_embeddings(geotiff_path: str) -> tuple[np.ndarray, dict]:
    """
    Load 64D Google Satellite Embeddings from GeoTIFF.

    Args:
        geotiff_path: Path to embedding GeoTIFF

    Returns:
        embeddings: (H, W, 64) float32 array
        meta: Dictionary with transform, crs, bounds, shape
    """
    print_progress(f"Loading embeddings from: {os.path.basename(geotiff_path)}")

    with rasterio.open(geotiff_path) as src:
        height, width = src.height, src.width
        n_bands = src.count

        if n_bands != 64:
            raise ValueError(f"Expected 64 bands, got {n_bands}")

        print_progress(f"Reading {n_bands} bands ({height} x {width} pixels)...")
        embeddings = np.zeros((height, width, 64), dtype=np.float32)

        for i in range(64):
            band = src.read(i + 1).astype(np.float32)
            band[np.isinf(band)] = 0
            embeddings[:, :, i] = band
            if (i + 1) % 16 == 0:
                print_progress(f"  Loaded bands 1-{i+1} of 64", indent=2)

        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'shape': (height, width),
            'resolution': src.res,
        }

    print_progress(f"Embeddings shape: {embeddings.shape}")

    return embeddings, meta


def compute_target_indices(geotiff_path: str) -> tuple[np.ndarray, dict]:
    """
    Compute NDVI, SAVI, BSI from Landsat 8/9 GeoTIFF.

    Args:
        geotiff_path: Path to Landsat GeoTIFF

    Returns:
        targets: (H, W, 3) float32 array with [NDVI, SAVI, BSI]
        meta: Dictionary with transform, crs, bounds, shape
    """
    print_progress(f"Computing target indices from: {os.path.basename(geotiff_path)}")

    with rasterio.open(geotiff_path) as src:
        print_progress("Reading Landsat bands...")
        blue = src.read(LANDSAT_BLUE_BAND).astype(np.float32)
        red = src.read(LANDSAT_RED_BAND).astype(np.float32)
        nir = src.read(LANDSAT_NIR_BAND).astype(np.float32)
        swir1 = src.read(LANDSAT_SWIR1_BAND).astype(np.float32)

        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'resolution': src.res,
        }

    # Compute indices
    print_progress("Computing NDVI...")
    ndvi = compute_ndvi(red, nir)

    print_progress("Computing SAVI (L=0.5)...")
    savi = compute_savi(red, nir, L=0.5)

    print_progress("Computing BSI...")
    bsi = compute_bsi(blue, red, nir, swir1)

    # Stack to (H, W, 3)
    targets = np.stack([ndvi, savi, bsi], axis=-1)

    print_progress(f"Target indices shape: {targets.shape}")
    for i, name in enumerate(TARGET_INDICES):
        valid = targets[:, :, i][np.isfinite(targets[:, :, i])]
        print_progress(f"  {name}: range=[{valid.min():.3f}, {valid.max():.3f}], mean={valid.mean():.3f}")

    return targets, meta


def resample_embeddings_to_target(
    embeddings: np.ndarray,
    source_meta: dict,
    target_meta: dict
) -> np.ndarray:
    """
    Downsample embeddings to match target grid (e.g., 10m -> 30m).

    Uses average resampling to preserve information.

    Args:
        embeddings: (H, W, 64) array
        source_meta: Dict with transform, crs, shape
        target_meta: Dict with transform, crs, shape

    Returns:
        Resampled embeddings matching target shape
    """
    target_shape = target_meta['shape']
    n_bands = embeddings.shape[2]

    print_progress(f"Downsampling embeddings from {embeddings.shape[:2]} to {target_shape}")

    resampled = np.zeros((*target_shape, n_bands), dtype=np.float32)

    start_time = time.time()
    for i in range(n_bands):
        destination = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=embeddings[:, :, i],
            destination=destination,
            src_transform=source_meta['transform'],
            src_crs=source_meta['crs'],
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            resampling=Resampling.average,
        )
        resampled[:, :, i] = destination

        if (i + 1) % 16 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_bands - i - 1)
            print_progress(f"  Resampled {i+1}/{n_bands} bands (ETA: {eta:.0f}s)", indent=2)

    print_progress(f"Resampling complete. New shape: {resampled.shape}")

    return resampled


# =============================================================================
# Multi-Index Projection
# =============================================================================

def fit_multi_index_projection(
    embeddings: np.ndarray,
    targets: np.ndarray,
    subsample_ratio: float = 0.1,
    alpha: float = 1.0,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit multi-output linear regression to find multiple directions in embedding space.

    Solves: [NDVI, SAVI, BSI] ≈ embeddings @ weights + biases

    Args:
        embeddings: (H, W, 64) embedding array
        targets: (H, W, 3) target indices array [NDVI, SAVI, BSI]
        subsample_ratio: Fraction of pixels for fitting
        alpha: Ridge regression regularization strength
        random_state: Random seed

    Returns:
        weights: (64, 3) weight matrix - each column is a projection direction
        biases: (3,) bias vector
        stats: Dictionary with R², correlation per index
    """
    print_progress("Preparing data for multi-output regression...")

    h, w, n_dim = embeddings.shape
    n_targets = targets.shape[2]

    # Flatten
    print_progress("Flattening arrays...")
    emb_flat = embeddings.reshape(-1, n_dim)
    targets_flat = targets.reshape(-1, n_targets)

    # Find valid pixels
    print_progress("Finding valid pixels...")
    valid_emb = np.any(emb_flat != 0, axis=1)
    valid_targets = np.all(np.isfinite(targets_flat), axis=1) & np.any(targets_flat != 0, axis=1)
    valid_mask = valid_emb & valid_targets

    X_valid = emb_flat[valid_mask]
    y_valid = targets_flat[valid_mask]

    print_progress(f"Valid pixels: {len(X_valid):,} / {h*w:,} ({100*len(X_valid)/(h*w):.1f}%)")

    if len(X_valid) < 1000:
        raise ValueError("Not enough valid pixels for regression")

    # Subsample for efficiency
    n_samples = max(10000, int(len(X_valid) * subsample_ratio))
    n_samples = min(n_samples, len(X_valid))

    print_progress(f"Subsampling {n_samples:,} pixels for fitting...")
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_valid), n_samples, replace=False)
    X_sample = X_valid[indices]
    y_sample = y_valid[indices]

    # Fit separate Ridge regression for each target
    # (could use MultiOutputRegressor but this gives clearer stats)
    print_progress(f"Fitting Ridge regression for {n_targets} targets (alpha={alpha})...")

    weights = np.zeros((n_dim, n_targets), dtype=np.float32)
    biases = np.zeros(n_targets, dtype=np.float32)
    stats = {'per_index': {}}

    for i, name in enumerate(TARGET_INDICES):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_sample, y_sample[:, i])

        weights[:, i] = model.coef_
        biases[i] = model.intercept_

        # Compute R² on sample
        y_pred = model.predict(X_sample)
        ss_res = np.sum((y_sample[:, i] - y_pred) ** 2)
        ss_tot = np.sum((y_sample[:, i] - y_sample[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Compute correlation
        correlation = np.corrcoef(y_sample[:, i], y_pred)[0, 1]

        stats['per_index'][name] = {
            'r2': float(r2),
            'correlation': float(correlation),
            'weight_norm': float(np.linalg.norm(weights[:, i])),
        }

        print_progress(f"  {name}: R²={r2:.4f}, corr={correlation:.4f}")

    # Overall stats
    stats['n_samples'] = n_samples
    stats['n_valid'] = len(X_valid)
    stats['alpha'] = alpha

    return weights, biases, stats


def project_to_multi_index(
    embeddings: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray
) -> np.ndarray:
    """
    Project embeddings to multi-dimensional index space.

    Args:
        embeddings: (H, W, 64) embedding array
        weights: (64, 3) weight matrix
        biases: (3,) bias vector

    Returns:
        projected: (H, W, 3) projected values [NDVI_proj, SAVI_proj, BSI_proj]
    """
    print_progress("Projecting embeddings to 3D index space...")

    # Matrix multiplication: (H, W, 64) @ (64, 3) -> (H, W, 3)
    projected = np.dot(embeddings, weights) + biases

    # Create validity mask
    print_progress("Applying validity mask...")
    valid_mask = np.any(embeddings != 0, axis=-1)
    projected[~valid_mask] = 0

    for i, name in enumerate(TARGET_INDICES):
        valid_proj = projected[:, :, i][valid_mask]
        print_progress(f"  {name}_proj: range=[{valid_proj.min():.4f}, {valid_proj.max():.4f}]")

    return projected


# =============================================================================
# K-Means Boundary Detection
# =============================================================================

def detect_boundary_kmeans(
    projected: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detect vegetation/desert boundary using K-means clustering in 3D space.

    Args:
        projected: (H, W, 3) projected values
        n_clusters: Number of clusters (default 2 for vegetation/desert)
        random_state: Random seed

    Returns:
        binary_mask: (H, W) bool array (True = vegetation)
        labels: (H, W) cluster labels
        vegetation_cluster: Which cluster ID is vegetation
    """
    print_progress(f"Running K-means clustering (k={n_clusters})...")

    h, w, n_dim = projected.shape

    # Flatten for clustering
    proj_flat = projected.reshape(-1, n_dim)

    # Valid pixels (non-zero projection)
    valid_mask = np.any(projected != 0, axis=-1).reshape(-1)
    X_valid = proj_flat[valid_mask]

    print_progress(f"Clustering {len(X_valid):,} valid pixels...")

    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_valid = kmeans.fit_predict(X_valid)

    # Create full label array
    labels_flat = np.full(h * w, -1, dtype=np.int32)
    labels_flat[valid_mask] = labels_valid
    labels = labels_flat.reshape(h, w)

    # Identify vegetation cluster (higher mean NDVI projection = index 0)
    cluster_means = []
    for c in range(n_clusters):
        mask = labels_valid == c
        if mask.sum() > 0:
            # Use first dimension (NDVI projection) to identify vegetation
            mean_ndvi = X_valid[mask, 0].mean()
            cluster_means.append((c, mean_ndvi))
            print_progress(f"  Cluster {c}: mean NDVI_proj={mean_ndvi:.4f}, count={mask.sum():,}")

    # Vegetation cluster has highest NDVI projection
    vegetation_cluster = max(cluster_means, key=lambda x: x[1])[0]
    print_progress(f"Identified vegetation cluster: {vegetation_cluster}")

    # Create binary mask
    binary_mask = (labels == vegetation_cluster)

    veg_pixels = binary_mask.sum()
    total_valid = valid_mask.sum()
    print_progress(f"Vegetation pixels: {veg_pixels:,} ({100*veg_pixels/total_valid:.1f}%)")

    return binary_mask, labels, vegetation_cluster


def threshold_single_index(
    values: np.ndarray,
    method: str = 'otsu'
) -> tuple[np.ndarray, float]:
    """
    Threshold a single index using Otsu's method.

    Args:
        values: (H, W) array
        method: Threshold method ('otsu')

    Returns:
        binary_mask: (H, W) bool array
        threshold: The threshold value used
    """
    valid_mask = values != 0
    valid_values = values[valid_mask]

    if method == 'otsu':
        vmin, vmax = valid_values.min(), valid_values.max()
        normalized = (valid_values - vmin) / (vmax - vmin + 1e-10)
        otsu_normalized = filters.threshold_otsu(normalized)
        threshold = otsu_normalized * (vmax - vmin) + vmin
    else:
        threshold = valid_values.mean()

    binary_mask = (values > threshold) & valid_mask

    return binary_mask, threshold


# =============================================================================
# Main Pipeline
# =============================================================================

def run_multi_index_pipeline(
    embeddings_path: str,
    landsat_path: str,
    alpha: float = 1.0,
    subsample_ratio: float = 0.1
) -> dict:
    """
    Full pipeline for multi-index boundary detection.

    Args:
        embeddings_path: Path to Google Satellite Embeddings GeoTIFF
        landsat_path: Path to Landsat GeoTIFF
        alpha: Ridge regularization strength
        subsample_ratio: Fraction of pixels for fitting

    Returns:
        Dictionary with all results
    """
    total_steps = 6
    start_time = time.time()

    print("\n" + "=" * 60)
    print("MULTI-INDEX BOUNDARY DETECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target indices: {TARGET_INDICES}")

    # Step 1: Load target indices
    print_step(1, total_steps, "Computing target indices from Landsat")
    targets, target_meta = compute_target_indices(landsat_path)

    # Step 2: Load embeddings
    print_step(2, total_steps, "Loading embeddings")
    embeddings_raw, emb_meta = load_embeddings(embeddings_path)

    # Step 3: Downsample embeddings to target resolution
    print_step(3, total_steps, "Downsampling embeddings to 30m resolution")
    if embeddings_raw.shape[:2] != targets.shape[:2]:
        embeddings = resample_embeddings_to_target(embeddings_raw, emb_meta, target_meta)
        working_meta = target_meta.copy()
    else:
        embeddings = embeddings_raw
        working_meta = emb_meta

    # Step 4: Fit multi-index projection
    print_step(4, total_steps, "Fitting multi-output Ridge regression")
    weights, biases, fit_stats = fit_multi_index_projection(
        embeddings, targets, subsample_ratio=subsample_ratio, alpha=alpha
    )

    # Step 5: Project embeddings
    print_step(5, total_steps, "Projecting embeddings to 3D index space")
    projected = project_to_multi_index(embeddings, weights, biases)

    # Step 6: K-means clustering
    print_step(6, total_steps, "K-means clustering for boundary detection")
    binary_mask, labels, veg_cluster = detect_boundary_kmeans(projected)

    # Build results
    elapsed = time.time() - start_time

    results = {
        # Projection parameters
        'weights': weights,
        'biases': biases,
        'fit_stats': fit_stats,
        # Projections
        'projected': projected,
        'targets': targets,
        # Clustering results
        'binary_mask': binary_mask,
        'labels': labels,
        'vegetation_cluster': veg_cluster,
        # Metadata
        'meta': working_meta,
        'target_indices': TARGET_INDICES,
        'metadata': {
            'embeddings_file': os.path.basename(embeddings_path),
            'landsat_file': os.path.basename(landsat_path),
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'alpha': alpha,
        }
    }

    print("\n" + "=" * 60)
    print("MULTI-INDEX PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"R² per index:")
    for name, stats in fit_stats['per_index'].items():
        print(f"  {name}: R²={stats['r2']:.4f}, corr={stats['correlation']:.4f}")
    print("=" * 60)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-index boundary detection using K-means in projected space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_index_boundary.py \\
      --embeddings data/google_embedding_2022.tif \\
      --landsat data/LC08_20221001_SAVI.tif

  python multi_index_boundary.py \\
      --embeddings data/google_embedding_2022.tif \\
      --landsat data/LC08_20221001_SAVI.tif \\
      --alpha 0.5
        """
    )

    parser.add_argument(
        '--embeddings', '-e',
        required=True,
        help='Path to Google Satellite Embeddings GeoTIFF (64 bands)'
    )

    parser.add_argument(
        '--landsat', '-l',
        required=True,
        help='Path to Landsat 8/9 GeoTIFF (for index calculation)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Ridge regularization strength (default: 1.0)'
    )

    parser.add_argument(
        '--subsample-ratio',
        type=float,
        default=0.1,
        help='Fraction of pixels for regression fitting (default: 0.1)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.embeddings):
        parser.error(f"Embeddings file not found: {args.embeddings}")
    if not os.path.exists(args.landsat):
        parser.error(f"Landsat file not found: {args.landsat}")

    # Run pipeline
    results = run_multi_index_pipeline(
        embeddings_path=args.embeddings,
        landsat_path=args.landsat,
        alpha=args.alpha,
        subsample_ratio=args.subsample_ratio
    )

    return results


if __name__ == '__main__':
    main()
