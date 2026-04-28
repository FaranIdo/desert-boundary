#!/usr/bin/env python3
"""
Boundary Detection Comparison: Embeddings vs NDVI Direct.

Compares two approaches for detecting desert-vegetation boundaries:
1. Embedding method: Google Embeddings → Ridge regression → Threshold → Boundary
2. NDVI baseline: NDVI directly → Threshold → Boundary

Outputs quantitative metrics (IoU, boundary agreement, smoothness) and visual comparison.

Usage:
    python analysis/boundary_comparison.py \
        --embeddings data/google_embedding_2022.tif \
        --landsat data/LC08_20221001_SAVI.tif

    # With name prefix to avoid overwriting:
    python analysis/boundary_comparison.py \
        --embeddings data/algeria/google_embedding_algeria_2023.tif \
        --landsat data/algeria/LC08_algeria_2023_composite_SAVI.tif \
        --name algeria
"""

import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime
from io import BytesIO
import base64

import numpy as np
from scipy import ndimage
from scipy.stats import linregress
from scipy.spatial import ConvexHull
import folium
from PIL import Image

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Import from boundary_detector
from analysis.boundary_detector import (
    compute_ndvi_from_landsat,
    threshold_projection,
    extract_main_boundary,
    load_embeddings,
    resample_embeddings_to_target,
    fit_ndvi_projection,
    project_embeddings,
    print_step,
    print_progress,
    create_boundary_visualization,
)

# Import from multi_index_boundary
from analysis.multi_index_boundary import (
    run_multi_index_pipeline,
    TARGET_INDICES,
)

# Paths
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
BOUNDARY_DIR = os.path.join(DATA_DIR, 'boundary')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')


# =============================================================================
# NDVI Boundary Detection (Baseline)
# =============================================================================

def compute_ndvi_boundary(
    landsat_path: str,
    threshold: float = None,
    smooth_kernel: int = 15,
    smooth_iterations: int = 5,
) -> dict:
    """
    Compute boundary using NDVI directly (baseline method).

    Args:
        landsat_path: Path to Landsat GeoTIFF
        threshold: Manual threshold (None for auto Otsu)
        smooth_kernel: Morphological kernel radius
        smooth_iterations: Number of smoothing iterations

    Returns:
        Dictionary with ndvi, smoothed_mask, boundary_mask, threshold, contours, meta
    """
    print_progress("Computing NDVI from Landsat...")
    ndvi, meta = compute_ndvi_from_landsat(landsat_path)

    print_progress("Thresholding NDVI directly...")
    binary_mask, used_threshold = threshold_projection(ndvi, threshold)

    print_progress("Extracting main boundary...")
    smoothed_mask, contours, boundary_mask = extract_main_boundary(
        binary_mask,
        heavy_smooth_kernel_size=smooth_kernel,
        heavy_smooth_iterations=smooth_iterations,
    )

    # Compute vegetation coverage
    valid_mask = ~np.isnan(ndvi) & (ndvi != 0)
    veg_fraction = np.sum(smoothed_mask & valid_mask) / np.sum(valid_mask)

    return {
        'ndvi': ndvi,
        'binary_mask': binary_mask,  # Raw mask before smoothing
        'smoothed_mask': smoothed_mask,
        'boundary_mask': boundary_mask,
        'threshold': used_threshold,
        'contours': contours,
        'meta': meta,
        'vegetation_fraction': veg_fraction,
        'method': 'ndvi_direct',
    }


def save_ndvi_boundary(result: dict, output_path: str):
    """
    Save NDVI boundary results to NPZ file.

    Args:
        result: Output from compute_ndvi_boundary()
        output_path: Path to save NPZ file
    """
    # Create boundary RGB visualization
    smoothed_mask = result['smoothed_mask']
    boundary_mask = result['boundary_mask']

    boundary_rgb = np.zeros((*smoothed_mask.shape, 3), dtype=np.uint8)
    boundary_rgb[smoothed_mask, 1] = 120  # Green for vegetation
    boundary_rgb[~smoothed_mask, 0] = 120  # Red component for desert
    boundary_rgb[~smoothed_mask, 1] = 80   # Some green for desert (brownish)
    boundary_rgb[boundary_mask, 0] = 0     # Blue boundary line
    boundary_rgb[boundary_mask, 1] = 0
    boundary_rgb[boundary_mask, 2] = 255

    np.savez_compressed(
        output_path,
        smoothed_mask=smoothed_mask.astype(np.uint8),
        boundary_mask=boundary_mask.astype(np.uint8),
        boundary_rgb=boundary_rgb,
        threshold=np.array([result['threshold']]),
        n_contours=np.array([len(result['contours'])]),
        metadata={
            'method': 'ndvi_direct',
            'timestamp': datetime.now().isoformat(),
            'vegetation_fraction': result['vegetation_fraction'],
        }
    )
    print_progress(f"NDVI boundary saved: {output_path}")


def save_embedding_boundary(result: dict, output_path: str):
    """
    Save embedding boundary results to NPZ file.

    Args:
        result: Output from compute_embedding_boundary()
        output_path: Path to save NPZ file
    """
    # Create boundary RGB visualization
    smoothed_mask = result['smoothed_mask']
    boundary_mask = result['boundary_mask']

    boundary_rgb = np.zeros((*smoothed_mask.shape, 3), dtype=np.uint8)
    boundary_rgb[smoothed_mask, 1] = 120  # Green for vegetation
    boundary_rgb[~smoothed_mask, 0] = 120  # Red component for desert
    boundary_rgb[~smoothed_mask, 1] = 80   # Some green for desert (brownish)
    boundary_rgb[boundary_mask, 0] = 0     # Blue boundary line
    boundary_rgb[boundary_mask, 1] = 0
    boundary_rgb[boundary_mask, 2] = 255

    # Get fit stats (r2, correlation) from result
    fit_stats = result.get('fit_stats', {})

    np.savez_compressed(
        output_path,
        smoothed_mask=smoothed_mask.astype(np.uint8),
        boundary_mask=boundary_mask.astype(np.uint8),
        boundary_rgb=boundary_rgb,
        projection=result.get('projection'),
        threshold=np.array([result['threshold']]),
        n_contours=np.array([len(result['contours'])]),
        fit_stats=fit_stats,
        metadata={
            'method': 'embedding',
            'timestamp': datetime.now().isoformat(),
            'vegetation_fraction': result['vegetation_fraction'],
            'r2': fit_stats.get('r2', 0),
            'correlation': fit_stats.get('correlation', 0),
        }
    )
    print_progress(f"Embedding boundary saved: {output_path}")


# =============================================================================
# Embedding Boundary Detection
# =============================================================================

def compute_embedding_boundary(
    embeddings_path: str,
    landsat_path: str,
    threshold: float = None,
    smooth_kernel: int = 15,
    smooth_iterations: int = 5,
    use_cached: bool = True,
) -> dict:
    """
    Compute boundary using embedding projection method.

    Args:
        embeddings_path: Path to embeddings GeoTIFF
        landsat_path: Path to Landsat GeoTIFF
        threshold: Manual threshold (None for auto Otsu)
        smooth_kernel: Morphological kernel radius
        smooth_iterations: Number of smoothing iterations
        use_cached: If True, load from existing NPZ if available

    Returns:
        Dictionary with projection, smoothed_mask, boundary_mask, threshold, contours, meta
    """
    cached_path = os.path.join(BOUNDARY_DIR, 'boundary_embedding.npz')

    if use_cached and os.path.exists(cached_path):
        print_progress(f"Loading cached embedding boundary: {cached_path}")
        data = np.load(cached_path, allow_pickle=True)

        # Load contours if available, otherwise empty list
        n_contours = int(data['n_contours'][0]) if 'n_contours' in data else 0

        # Reconstruct binary_mask from projection and threshold if not cached
        projection = data['projection'] if 'projection' in data else None
        threshold = float(data['threshold'][0])
        if projection is not None:
            valid_mask = projection != 0
            binary_mask = (projection > threshold) & valid_mask
        else:
            binary_mask = None

        return {
            'projection': projection,
            'binary_mask': binary_mask,  # Raw mask before smoothing
            'smoothed_mask': data['smoothed_mask'].astype(bool),
            'boundary_mask': data['boundary_mask'].astype(bool),
            'threshold': threshold,
            'contours': [],  # Not stored in NPZ
            'meta': data['metadata'].item() if 'metadata' in data else {},
            'vegetation_fraction': np.mean(data['smoothed_mask']),
            'method': 'embedding',
            'fit_stats': data['fit_stats'].item() if 'fit_stats' in data else {},
        }

    # Run full pipeline
    print_progress("Running full embedding pipeline...")

    # Load NDVI
    ndvi, ndvi_meta = compute_ndvi_from_landsat(landsat_path)

    # Load embeddings
    embeddings_raw, emb_meta = load_embeddings(embeddings_path)

    # Resample embeddings to NDVI resolution
    if embeddings_raw.shape[:2] != ndvi.shape:
        embeddings = resample_embeddings_to_target(embeddings_raw, emb_meta, ndvi_meta)
    else:
        embeddings = embeddings_raw

    # Fit projection
    weights, bias, fit_stats = fit_ndvi_projection(embeddings, ndvi)

    # Project
    projection = project_embeddings(embeddings, weights, bias)

    # Threshold
    binary_mask, used_threshold = threshold_projection(projection, threshold)

    # Extract boundary
    smoothed_mask, contours, boundary_mask = extract_main_boundary(
        binary_mask,
        heavy_smooth_kernel_size=smooth_kernel,
        heavy_smooth_iterations=smooth_iterations,
    )

    valid_mask = ~np.isnan(projection) & (projection != 0)
    veg_fraction = np.sum(smoothed_mask & valid_mask) / np.sum(valid_mask)

    return {
        'projection': projection,
        'binary_mask': binary_mask,  # Raw mask before smoothing
        'smoothed_mask': smoothed_mask,
        'boundary_mask': boundary_mask,
        'threshold': used_threshold,
        'contours': contours,
        'meta': ndvi_meta,
        'vegetation_fraction': veg_fraction,
        'method': 'embedding',
        'fit_stats': fit_stats,
    }


# =============================================================================
# Comparison Metrics
# =============================================================================

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        IoU value in [0, 1]
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_boundary_agreement(
    boundary1: np.ndarray,
    boundary2: np.ndarray,
    tolerance_pixels: int = 2,
) -> dict:
    """
    Compute boundary agreement with spatial tolerance.

    Args:
        boundary1: First boundary mask
        boundary2: Second boundary mask
        tolerance_pixels: Distance tolerance for matching

    Returns:
        Dictionary with agreement metrics
    """
    # Distance transform of boundary2
    dist2 = ndimage.distance_transform_edt(~boundary2)

    # Count boundary1 pixels within tolerance of boundary2
    boundary1_pixels = np.sum(boundary1)
    if boundary1_pixels > 0:
        within_tolerance_1_to_2 = np.sum(boundary1 & (dist2 <= tolerance_pixels))
        agreement_1_to_2 = within_tolerance_1_to_2 / boundary1_pixels
    else:
        agreement_1_to_2 = 0.0

    # Distance transform of boundary1
    dist1 = ndimage.distance_transform_edt(~boundary1)

    # Count boundary2 pixels within tolerance of boundary1
    boundary2_pixels = np.sum(boundary2)
    if boundary2_pixels > 0:
        within_tolerance_2_to_1 = np.sum(boundary2 & (dist1 <= tolerance_pixels))
        agreement_2_to_1 = within_tolerance_2_to_1 / boundary2_pixels
    else:
        agreement_2_to_1 = 0.0

    return {
        'agreement_1_to_2': agreement_1_to_2,
        'agreement_2_to_1': agreement_2_to_1,
        'mean_agreement': (agreement_1_to_2 + agreement_2_to_1) / 2,
        'tolerance_pixels': tolerance_pixels,
    }


def compute_curvature(contour: np.ndarray) -> np.ndarray:
    """
    Compute local curvature at each point of a contour.

    Args:
        contour: (N, 2) array of contour points

    Returns:
        (N,) array of curvature values
    """
    if len(contour) < 3:
        return np.array([0.0])

    # Compute vectors between consecutive points
    v1 = np.diff(contour, axis=0)  # vectors to next point
    v1 = np.vstack([v1, contour[0] - contour[-1]])  # close the loop

    v2 = np.roll(v1, -1, axis=0)  # vectors from current to next-next

    # Compute magnitudes
    mag1 = np.linalg.norm(v1, axis=1) + 1e-10
    mag2 = np.linalg.norm(v2, axis=1) + 1e-10

    # Compute angles
    cos_angles = np.sum(v1 * v2, axis=1) / (mag1 * mag2)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)

    # Curvature = angle / average segment length
    avg_length = (mag1 + mag2) / 2
    curvature = angles / avg_length

    return curvature


def compute_fractal_dimension(boundary_mask: np.ndarray) -> float:
    """
    Estimate fractal dimension using box-counting method.

    Args:
        boundary_mask: Binary boundary mask

    Returns:
        Estimated fractal dimension
    """
    # Get boundary coordinates
    coords = np.argwhere(boundary_mask)

    if len(coords) < 10:
        return 1.0

    # Determine box sizes
    min_dim = min(boundary_mask.shape)
    box_sizes = []
    size = 4
    while size < min_dim // 2:
        box_sizes.append(size)
        size *= 2

    if len(box_sizes) < 3:
        return 1.0

    counts = []
    for box_size in box_sizes:
        # Count occupied boxes
        boxes = set()
        for r, c in coords:
            box_r = r // box_size
            box_c = c // box_size
            boxes.add((box_r, box_c))
        counts.append(len(boxes))

    # Linear regression on log-log scale
    log_sizes = np.log(1 / np.array(box_sizes))
    log_counts = np.log(np.array(counts))

    slope, _, _, _, _ = linregress(log_sizes, log_counts)

    return slope


def compute_sinuosity(contour: np.ndarray) -> float:
    """
    Compute sinuosity of a contour.

    Args:
        contour: (N, 2) array of contour points

    Returns:
        Sinuosity value (1.0 = straight, higher = more winding)
    """
    if len(contour) < 3:
        return 1.0

    # Total path length
    diffs = np.diff(contour, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # For closed contour, use convex hull perimeter as reference
    try:
        hull = ConvexHull(contour)
        hull_perimeter = 0
        for i in range(len(hull.vertices)):
            p1 = contour[hull.vertices[i]]
            p2 = contour[hull.vertices[(i + 1) % len(hull.vertices)]]
            hull_perimeter += np.linalg.norm(p2 - p1)

        if hull_perimeter > 0:
            return path_length / hull_perimeter
    except:
        pass

    # Fallback: straight line distance
    straight_dist = np.linalg.norm(contour[-1] - contour[0])
    if straight_dist > 0:
        return path_length / straight_dist

    return 1.0


def compute_boundary_smoothness(contours: list, boundary_mask: np.ndarray) -> dict:
    """
    Compute smoothness metrics for boundary.

    Args:
        contours: List of contour arrays
        boundary_mask: Binary boundary mask

    Returns:
        Dictionary with smoothness metrics
    """
    if not contours:
        return {
            'curvature_variance': 0.0,
            'curvature_mean': 0.0,
            'fractal_dimension': 1.0,
            'sinuosity': 1.0,
            'total_length': 0,
        }

    # Use longest contour for curvature and sinuosity
    main_contour = max(contours, key=len)

    # Curvature
    curvature = compute_curvature(main_contour)
    curvature_variance = np.var(curvature)
    curvature_mean = np.mean(np.abs(curvature))

    # Fractal dimension (from full boundary mask)
    fractal_dim = compute_fractal_dimension(boundary_mask)

    # Sinuosity
    sinuosity = compute_sinuosity(main_contour)

    # Total boundary length
    total_length = sum(len(c) for c in contours)

    return {
        'curvature_variance': float(curvature_variance),
        'curvature_mean': float(curvature_mean),
        'fractal_dimension': float(fractal_dim),
        'sinuosity': float(sinuosity),
        'total_length': total_length,
    }


# =============================================================================
# Main Comparison
# =============================================================================

def extract_contours_from_mask(boundary_mask: np.ndarray) -> list:
    """Extract contours from boundary mask using skimage."""
    from skimage import measure
    contours = measure.find_contours(boundary_mask.astype(np.uint8), level=0.5)
    return sorted(contours, key=len, reverse=True)


def compare_boundaries(
    embedding_result: dict,
    ndvi_result: dict,
    tolerance_pixels: int = 2,
) -> dict:
    """
    Compare embedding and NDVI boundary detection results.

    Args:
        embedding_result: Output from compute_embedding_boundary()
        ndvi_result: Output from compute_ndvi_boundary()
        tolerance_pixels: Tolerance for boundary agreement

    Returns:
        Dictionary with all comparison metrics
    """
    # Mask IoU
    iou = compute_mask_iou(
        embedding_result['smoothed_mask'],
        ndvi_result['smoothed_mask'],
    )

    # Boundary agreement
    agreement = compute_boundary_agreement(
        embedding_result['boundary_mask'],
        ndvi_result['boundary_mask'],
        tolerance_pixels=tolerance_pixels,
    )

    # Extract contours from boundary masks if not available
    embedding_contours = embedding_result.get('contours', [])
    if not embedding_contours:
        embedding_contours = extract_contours_from_mask(embedding_result['boundary_mask'])

    ndvi_contours = ndvi_result.get('contours', [])
    if not ndvi_contours:
        ndvi_contours = extract_contours_from_mask(ndvi_result['boundary_mask'])

    # Smoothness metrics
    smoothness_embedding = compute_boundary_smoothness(
        embedding_contours,
        embedding_result['boundary_mask'],
    )

    smoothness_ndvi = compute_boundary_smoothness(
        ndvi_contours,
        ndvi_result['boundary_mask'],
    )

    # Summary
    curv_diff = smoothness_ndvi['curvature_variance'] - smoothness_embedding['curvature_variance']
    if smoothness_ndvi['curvature_variance'] > 0:
        curv_diff_pct = curv_diff / smoothness_ndvi['curvature_variance'] * 100
    else:
        curv_diff_pct = 0

    # Convert all values to native Python types for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        return obj

    return to_native({
        'iou': iou,
        'boundary_agreement': agreement,
        'smoothness_embedding': smoothness_embedding,
        'smoothness_ndvi': smoothness_ndvi,
        'vegetation_coverage': {
            'embedding': embedding_result['vegetation_fraction'],
            'ndvi': ndvi_result['vegetation_fraction'],
            'difference': abs(embedding_result['vegetation_fraction'] - ndvi_result['vegetation_fraction']),
        },
        'thresholds': {
            'embedding': embedding_result['threshold'],
            'ndvi': ndvi_result['threshold'],
        },
        'summary': {
            'embedding_smoother_by_pct': curv_diff_pct,
            'masks_similar': iou > 0.8,
            'boundaries_similar': agreement['mean_agreement'] > 0.7,
        },
    })


def create_agreement_overlay(
    embedding_boundary: np.ndarray,
    ndvi_boundary: np.ndarray,
    embedding_mask: np.ndarray,
    ndvi_mask: np.ndarray,
) -> np.ndarray:
    """
    Create RGB overlay showing boundary agreement/disagreement.

    Args:
        embedding_boundary: Embedding boundary mask
        ndvi_boundary: NDVI boundary mask
        embedding_mask: Embedding vegetation mask
        ndvi_mask: NDVI vegetation mask

    Returns:
        (H, W, 3) RGB array
    """
    h, w = embedding_boundary.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Background: light gray for agree, colors for disagree
    both_veg = embedding_mask & ndvi_mask
    both_desert = ~embedding_mask & ~ndvi_mask
    agree = both_veg | both_desert

    # Where masks disagree
    embedding_only_veg = embedding_mask & ~ndvi_mask  # Embedding sees veg, NDVI doesn't
    ndvi_only_veg = ndvi_mask & ~embedding_mask       # NDVI sees veg, Embedding doesn't

    overlay[agree] = [200, 200, 200]       # Gray for agreement
    overlay[embedding_only_veg] = [255, 200, 200]  # Light red
    overlay[ndvi_only_veg] = [200, 200, 255]       # Light blue

    # Boundary overlay
    both_boundary = embedding_boundary & ndvi_boundary
    embedding_only_boundary = embedding_boundary & ~ndvi_boundary
    ndvi_only_boundary = ndvi_boundary & ~embedding_boundary

    overlay[both_boundary] = [0, 200, 0]          # Green: both detect
    overlay[embedding_only_boundary] = [255, 0, 0]  # Red: embedding only
    overlay[ndvi_only_boundary] = [0, 0, 255]       # Blue: NDVI only

    return overlay


# =============================================================================
# Visualization
# =============================================================================

def create_raw_comparison_visualization(
    embedding_result: dict,
    ndvi_result: dict,
    output_path: str,
) -> str:
    """
    Create HTML visualization showing RAW outputs before post-processing.

    Shows:
    - Continuous heatmaps: embedding projection vs raw NDVI
    - Unsmoothed binary masks: after Otsu threshold, before morphological smoothing

    Args:
        embedding_result: Embedding boundary result with 'projection' and 'binary_mask'
        ndvi_result: NDVI boundary result with 'ndvi' and 'binary_mask'
        output_path: Output HTML path

    Returns:
        HTML content
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    meta = embedding_result.get('meta', ndvi_result.get('meta', {}))
    bounds = meta.get('bounds', None)

    if bounds:
        center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
        leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    else:
        center = [31.25, 34.8]
        leaflet_bounds = [[31.0, 34.5], [31.5, 35.1]]

    # Create map
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # Add basemap
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='&copy; Esri',
        name='Satellite',
    ).add_to(m)

    # Helper to create colormap image
    def create_heatmap_rgba(data: np.ndarray, cmap_name: str = 'viridis', vmin: float = None, vmax: float = None) -> tuple[np.ndarray, float, float]:
        """Convert continuous data to RGBA using percentile normalization.

        Returns:
            rgba: RGBA image array
            vmin: Minimum value used for normalization
            vmax: Maximum value used for normalization
        """
        valid_mask = (data != 0) & ~np.isnan(data)
        if not np.any(valid_mask):
            return np.zeros((*data.shape, 4), dtype=np.uint8), 0, 1

        # Percentile normalization (2-98%) if not provided
        valid_values = data[valid_mask]
        if vmin is None:
            vmin = np.percentile(valid_values, 2)
        if vmax is None:
            vmax = np.percentile(valid_values, 98)

        # Normalize
        normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)

        # Apply colormap
        cmap = plt.get_cmap(cmap_name)
        rgba = (cmap(normalized) * 255).astype(np.uint8)

        # Set alpha: transparent where invalid
        rgba[~valid_mask, 3] = 0
        rgba[valid_mask, 3] = 200

        return rgba, vmin, vmax

    def create_binary_rgba(mask: np.ndarray, veg_color=(0, 150, 0), desert_color=(150, 100, 50)) -> np.ndarray:
        """Convert binary mask to RGBA (green=veg, brown=desert)."""
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)

        # Vegetation (True)
        rgba[mask, 0] = veg_color[0]
        rgba[mask, 1] = veg_color[1]
        rgba[mask, 2] = veg_color[2]
        rgba[mask, 3] = 150

        # Desert (False)
        rgba[~mask, 0] = desert_color[0]
        rgba[~mask, 1] = desert_color[1]
        rgba[~mask, 2] = desert_color[2]
        rgba[~mask, 3] = 100

        return rgba

    def add_image_overlay(m, rgba_data: np.ndarray, bounds, name: str):
        """Add RGBA image as overlay layer."""
        img = Image.fromarray(rgba_data)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_base64}",
            bounds=bounds,
            opacity=0.7,
            name=name,
        ).add_to(m)

    # Use same colormap for both continuous layers for easy comparison
    # Custom brown-to-green colormap (desert=brown, vegetation=green)
    from matplotlib.colors import LinearSegmentedColormap
    brown_green_colors = ['#8B4513', '#A0522D', '#CD853F', '#D2B48C', '#F5DEB3',
                          '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
    CONTINUOUS_CMAP = LinearSegmentedColormap.from_list('BrownGreen', brown_green_colors, N=256)

    # Track value ranges for legend
    proj_vmin, proj_vmax = 0, 1
    ndvi_vmin, ndvi_vmax = 0, 1

    # Layer 1: Embedding projection heatmap
    if embedding_result.get('projection') is not None:
        proj_rgba, proj_vmin, proj_vmax = create_heatmap_rgba(embedding_result['projection'], CONTINUOUS_CMAP)
        add_image_overlay(m, proj_rgba, leaflet_bounds, 'Embedding Projection (continuous)')

    # Layer 2: NDVI heatmap
    if ndvi_result.get('ndvi') is not None:
        ndvi_rgba, ndvi_vmin, ndvi_vmax = create_heatmap_rgba(ndvi_result['ndvi'], CONTINUOUS_CMAP)
        add_image_overlay(m, ndvi_rgba, leaflet_bounds, 'Raw NDVI (continuous)')

    # Layer 3: Embedding binary mask (unsmoothed)
    if embedding_result.get('binary_mask') is not None:
        emb_binary_rgba = create_binary_rgba(embedding_result['binary_mask'])
        add_image_overlay(m, emb_binary_rgba, leaflet_bounds, 'Embedding Binary (unsmoothed)')

    # Layer 4: NDVI binary mask (unsmoothed)
    if ndvi_result.get('binary_mask') is not None:
        ndvi_binary_rgba = create_binary_rgba(ndvi_result['binary_mask'], veg_color=(0, 100, 200))
        add_image_overlay(m, ndvi_binary_rgba, leaflet_bounds, 'NDVI Binary (unsmoothed)')

    # Add layer control
    folium.LayerControl().add_to(m)

    # Compute raw comparison stats
    emb_thresh = embedding_result.get('threshold', 0)
    ndvi_thresh = ndvi_result.get('threshold', 0)

    emb_binary = embedding_result.get('binary_mask')
    ndvi_binary = ndvi_result.get('binary_mask')

    if emb_binary is not None and ndvi_binary is not None:
        # Raw IoU (before smoothing)
        intersection = np.sum(emb_binary & ndvi_binary)
        union = np.sum(emb_binary | ndvi_binary)
        raw_iou = intersection / union if union > 0 else 0

        # Raw vegetation coverage
        emb_veg_pct = np.mean(emb_binary) * 100
        ndvi_veg_pct = np.mean(ndvi_binary) * 100
    else:
        raw_iou = 0
        emb_veg_pct = 0
        ndvi_veg_pct = 0

    # Correlation between continuous values
    proj = embedding_result.get('projection')
    ndvi = ndvi_result.get('ndvi')
    if proj is not None and ndvi is not None:
        valid = (proj != 0) & ~np.isnan(proj) & (ndvi != 0) & ~np.isnan(ndvi)
        if np.sum(valid) > 100:
            corr = np.corrcoef(proj[valid].flatten(), ndvi[valid].flatten())[0, 1]
        else:
            corr = 0
    else:
        corr = 0

    # R² from fit stats
    fit_stats = embedding_result.get('fit_stats', {})
    r2 = fit_stats.get('r2', 0)

    # Legend/metrics panel with colorbar
    # Brown-to-green gradient CSS (desert=brown, vegetation=green)
    brown_green_gradient = "linear-gradient(to right, #8B4513, #A0522D, #CD853F, #D2B48C, #F5DEB3, #ADFF2F, #7CFC00, #32CD32, #228B22, #006400)"

    legend_html = f'''
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 400px;
    ">
        <h4 style="margin: 0 0 10px 0;">Raw Output Comparison (Before Post-Processing)</h4>

        <div style="margin-bottom: 10px; padding: 8px; background: #f5f5f5; border-radius: 4px;">
            <strong>Key Question:</strong> Does embedding provide better signal than raw NDVI?
        </div>

        <!-- Colorbar for continuous layers -->
        <div style="margin-bottom: 12px; padding: 10px; background: #fafafa; border-radius: 4px;">
            <strong>Continuous Value Scale</strong>
            <div style="
                height: 20px;
                background: {brown_green_gradient};
                border-radius: 3px;
                margin: 8px 0 4px 0;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666;">
                <span>Low (desert)</span>
                <span>High (vegetation)</span>
            </div>

            <table style="width: 100%; font-size: 10px; margin-top: 8px; border-collapse: collapse;">
                <tr style="background: #e8e8e8;">
                    <th style="padding: 3px; text-align: left;">Layer</th>
                    <th style="padding: 3px;">Min</th>
                    <th style="padding: 3px;">Max</th>
                    <th style="padding: 3px;">Threshold</th>
                </tr>
                <tr>
                    <td style="padding: 3px;">Embedding Proj</td>
                    <td style="padding: 3px; text-align: center;">{proj_vmin:.3f}</td>
                    <td style="padding: 3px; text-align: center;">{proj_vmax:.3f}</td>
                    <td style="padding: 3px; text-align: center;">{emb_thresh:.3f}</td>
                </tr>
                <tr style="background: #f5f5f5;">
                    <td style="padding: 3px;">Raw NDVI</td>
                    <td style="padding: 3px; text-align: center;">{ndvi_vmin:.3f}</td>
                    <td style="padding: 3px; text-align: center;">{ndvi_vmax:.3f}</td>
                    <td style="padding: 3px; text-align: center;">{ndvi_thresh:.3f}</td>
                </tr>
            </table>
        </div>

        <!-- Binary masks legend -->
        <div style="margin-bottom: 10px;">
            <strong>Binary Masks (unsmoothed)</strong>
            <div style="margin-top: 5px; display: flex; gap: 15px;">
                <div>
                    <span style="display: inline-block; width: 14px; height: 14px; background: rgb(0,150,0); vertical-align: middle; border-radius: 2px;"></span>
                    <span style="font-size: 11px;"> Emb veg ({emb_veg_pct:.1f}%)</span>
                </div>
                <div>
                    <span style="display: inline-block; width: 14px; height: 14px; background: rgb(0,100,200); vertical-align: middle; border-radius: 2px;"></span>
                    <span style="font-size: 11px;"> NDVI veg ({ndvi_veg_pct:.1f}%)</span>
                </div>
            </div>
        </div>

        <div style="padding-top: 8px; border-top: 1px solid #ccc;">
            <strong>Comparison Metrics:</strong><br>
            <span style="margin-left: 8px;">• Raw Mask IoU: <strong>{raw_iou:.1%}</strong></span><br>
            <span style="margin-left: 8px;">• Correlation (proj↔NDVI): <strong>{corr:.3f}</strong></span><br>
            <span style="margin-left: 8px;">• Ridge R²: <strong>{r2:.3f}</strong></span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    html = m._repr_html_()

    with open(output_path, 'w') as f:
        f.write(html)

    print_progress(f"Raw comparison visualization saved: {output_path}")

    return html


def create_sidebyside_raw_viewer(
    embedding_result: dict,
    ndvi_result: dict,
    output_path: str,
    multi_index_result: dict = None,
) -> str:
    """
    Create side-by-side viewer for raw outputs (before post-processing).

    Allows selecting which layer to show on left and right panels.
    Optionally includes multi-index (NDVI+SAVI+BSI) results for three-way comparison.

    Args:
        embedding_result: Embedding boundary result with 'projection' and 'binary_mask'
        ndvi_result: NDVI boundary result with 'ndvi' and 'binary_mask'
        output_path: Output HTML path
        multi_index_result: Optional multi-index result with 'projected' and 'binary_mask'

    Returns:
        HTML content
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    meta = embedding_result.get('meta', ndvi_result.get('meta', {}))
    bounds = meta.get('bounds', None)

    if bounds:
        center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
        leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    else:
        center = [31.25, 34.8]
        leaflet_bounds = [[31.0, 34.5], [31.5, 35.1]]

    # Create brown-to-green colormap
    brown_green_colors = ['#8B4513', '#A0522D', '#CD853F', '#D2B48C', '#F5DEB3',
                          '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
    brown_green_cmap = LinearSegmentedColormap.from_list('BrownGreen', brown_green_colors, N=256)

    def create_heatmap_rgba(data: np.ndarray, cmap) -> tuple[np.ndarray, float, float]:
        """Convert continuous data to RGBA."""
        valid_mask = (data != 0) & ~np.isnan(data)
        if not np.any(valid_mask):
            return np.zeros((*data.shape, 4), dtype=np.uint8), 0, 1

        valid_values = data[valid_mask]
        vmin = np.percentile(valid_values, 2)
        vmax = np.percentile(valid_values, 98)

        normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
        rgba = (cmap(normalized) * 255).astype(np.uint8)
        rgba[~valid_mask, 3] = 0
        rgba[valid_mask, 3] = 200

        return rgba, vmin, vmax

    def create_binary_rgba(mask: np.ndarray, veg_color, desert_color) -> np.ndarray:
        """Convert binary mask to RGBA."""
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask, 0] = veg_color[0]
        rgba[mask, 1] = veg_color[1]
        rgba[mask, 2] = veg_color[2]
        rgba[mask, 3] = 180
        rgba[~mask, 0] = desert_color[0]
        rgba[~mask, 1] = desert_color[1]
        rgba[~mask, 2] = desert_color[2]
        rgba[~mask, 3] = 120
        return rgba

    def rgba_to_base64(rgba: np.ndarray) -> str:
        """Convert RGBA array to base64 PNG."""
        img = Image.fromarray(rgba)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    # Render all layers
    images = {}
    layer_info = {}

    # Embedding projection
    if embedding_result.get('projection') is not None:
        rgba, vmin, vmax = create_heatmap_rgba(embedding_result['projection'], brown_green_cmap)
        images['emb_proj'] = rgba_to_base64(rgba)
        layer_info['emb_proj'] = {
            'name': 'Embedding Projection',
            'type': 'continuous',
            'vmin': f'{vmin:.3f}',
            'vmax': f'{vmax:.3f}',
            'threshold': f'{embedding_result.get("threshold", 0):.3f}',
        }

    # Raw NDVI
    if ndvi_result.get('ndvi') is not None:
        rgba, vmin, vmax = create_heatmap_rgba(ndvi_result['ndvi'], brown_green_cmap)
        images['ndvi_raw'] = rgba_to_base64(rgba)
        layer_info['ndvi_raw'] = {
            'name': 'Raw NDVI',
            'type': 'continuous',
            'vmin': f'{vmin:.3f}',
            'vmax': f'{vmax:.3f}',
            'threshold': f'{ndvi_result.get("threshold", 0):.3f}',
        }

    # Embedding binary (unsmoothed)
    if embedding_result.get('binary_mask') is not None:
        rgba = create_binary_rgba(embedding_result['binary_mask'], (0, 150, 0), (139, 69, 19))
        images['emb_binary'] = rgba_to_base64(rgba)
        veg_pct = np.mean(embedding_result['binary_mask']) * 100
        layer_info['emb_binary'] = {
            'name': 'Embedding Binary (unsmoothed)',
            'type': 'binary',
            'veg_coverage': f'{veg_pct:.1f}%',
            'threshold': f'{embedding_result.get("threshold", 0):.3f}',
        }

    # NDVI binary (unsmoothed)
    if ndvi_result.get('binary_mask') is not None:
        rgba = create_binary_rgba(ndvi_result['binary_mask'], (0, 100, 200), (139, 69, 19))
        images['ndvi_binary'] = rgba_to_base64(rgba)
        veg_pct = np.mean(ndvi_result['binary_mask']) * 100
        layer_info['ndvi_binary'] = {
            'name': 'NDVI Binary (unsmoothed)',
            'type': 'binary',
            'veg_coverage': f'{veg_pct:.1f}%',
            'threshold': f'{ndvi_result.get("threshold", 0):.3f}',
        }

    # Multi-index projection (average of 3D for visualization)
    if multi_index_result is not None and multi_index_result.get('projected') is not None:
        # Use mean of 3 projections for continuous visualization
        multi_proj_mean = np.mean(multi_index_result['projected'], axis=-1)
        rgba, vmin, vmax = create_heatmap_rgba(multi_proj_mean, brown_green_cmap)
        images['multi_proj'] = rgba_to_base64(rgba)

        # Get R² stats per index
        fit_stats = multi_index_result.get('fit_stats', {})
        r2_str = ', '.join([
            f"{name}={fit_stats['per_index'][name]['r2']:.2f}"
            for name in TARGET_INDICES
            if name in fit_stats.get('per_index', {})
        ])

        layer_info['multi_proj'] = {
            'name': 'Multi-Index Projection (mean)',
            'type': 'continuous',
            'vmin': f'{vmin:.3f}',
            'vmax': f'{vmax:.3f}',
            'r2_per_index': r2_str,
        }

    # Multi-index binary (K-means)
    if multi_index_result is not None and multi_index_result.get('binary_mask') is not None:
        rgba = create_binary_rgba(multi_index_result['binary_mask'], (255, 165, 0), (139, 69, 19))  # Orange for multi-index
        images['multi_binary'] = rgba_to_base64(rgba)
        veg_pct = np.mean(multi_index_result['binary_mask']) * 100
        layer_info['multi_binary'] = {
            'name': 'Multi-Index Binary (K-means)',
            'type': 'binary',
            'veg_coverage': f'{veg_pct:.1f}%',
            'method': 'K-means (k=2)',
        }

    # Compute comparison stats
    emb_binary = embedding_result.get('binary_mask')
    ndvi_binary = ndvi_result.get('binary_mask')
    multi_binary = multi_index_result.get('binary_mask') if multi_index_result else None

    def compute_iou(mask1, mask2):
        if mask1 is None or mask2 is None:
            return 0
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        return intersection / union if union > 0 else 0

    def compute_correlation(arr1, arr2):
        if arr1 is None or arr2 is None:
            return 0
        valid = (arr1 != 0) & (arr2 != 0) & np.isfinite(arr1) & np.isfinite(arr2)
        if np.sum(valid) < 100:
            return 0
        return np.corrcoef(arr1[valid].flatten(), arr2[valid].flatten())[0, 1]

    # IoU matrix
    iou_ndvi_emb = compute_iou(ndvi_binary, emb_binary)
    iou_ndvi_multi = compute_iou(ndvi_binary, multi_binary)
    iou_emb_multi = compute_iou(emb_binary, multi_binary)

    # Correlation matrix (continuous values)
    emb_proj = embedding_result.get('projection')
    ndvi_raw = ndvi_result.get('ndvi')
    multi_proj_mean = np.mean(multi_index_result['projected'], axis=-1) if multi_index_result and multi_index_result.get('projected') is not None else None

    corr_ndvi_emb = compute_correlation(ndvi_raw, emb_proj)
    corr_ndvi_multi = compute_correlation(ndvi_raw, multi_proj_mean)
    corr_emb_multi = compute_correlation(emb_proj, multi_proj_mean)

    # R² values
    fit_stats = embedding_result.get('fit_stats', {})
    r2_emb = fit_stats.get('r2', 0)
    corr_emb = fit_stats.get('correlation', 0)

    # Multi-index R² per index
    multi_r2 = {}
    if multi_index_result:
        multi_fit_stats = multi_index_result.get('fit_stats', {})
        for name in TARGET_INDICES:
            if name in multi_fit_stats.get('per_index', {}):
                multi_r2[name] = multi_fit_stats['per_index'][name]['r2']

    # Legacy variables for backwards compatibility
    raw_iou = iou_ndvi_emb
    r2 = r2_emb
    corr = corr_emb

    # Brown-to-green gradient CSS
    gradient_css = "linear-gradient(to right, #8B4513, #A0522D, #CD853F, #D2B48C, #F5DEB3, #ADFF2F, #7CFC00, #32CD32, #228B22, #006400)"

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Raw Boundary Comparison - Side by Side</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; }}

        .container {{ display: flex; height: 100vh; }}

        .map-panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 2px solid #333;
        }}
        .map-panel:last-child {{ border-right: none; }}

        .panel-header {{
            background: #16213e;
            padding: 10px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .panel-header select {{
            flex: 1;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #444;
            border-radius: 4px;
            background: #0f3460;
            color: white;
            cursor: pointer;
        }}
        .panel-header select:hover {{ border-color: #667; }}

        .map {{ flex: 1; }}

        .panel-info {{
            background: #16213e;
            padding: 10px 15px;
            color: #ccc;
            font-size: 12px;
            min-height: 80px;
        }}
        .panel-info .title {{ font-weight: bold; color: white; margin-bottom: 6px; }}
        .panel-info .stat {{ margin: 3px 0; }}
        .panel-info .stat-label {{ color: #888; }}

        #controls {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 33, 62, 0.95);
            padding: 12px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 15px;
            z-index: 1000;
            color: white;
            font-size: 13px;
        }}
        #controls input[type="range"] {{ width: 120px; }}

        #legend {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 33, 62, 0.95);
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 1000;
            color: white;
            font-size: 12px;
            text-align: center;
        }}
        #legend .gradient {{
            height: 16px;
            width: 200px;
            background: {gradient_css};
            border-radius: 3px;
            margin: 8px 0;
        }}
        #legend .labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #aaa; }}
        #legend .metrics {{ margin-top: 10px; font-size: 11px; color: #ccc; }}
        #legend table {{ border-collapse: collapse; margin: 8px auto; font-size: 10px; }}
        #legend th, #legend td {{ padding: 4px 8px; border: 1px solid #444; text-align: center; }}
        #legend th {{ background: #0f3460; }}
        #legend .section-title {{ font-weight: bold; margin-top: 10px; margin-bottom: 5px; color: #aaa; }}
    </style>
</head>
<body>
    <div id="legend">
        <strong>Continuous Value Scale</strong>
        <div class="gradient"></div>
        <div class="labels"><span>Desert (low)</span><span>Vegetation (high)</span></div>
        {'<div class="section-title">IoU Matrix (Binary Masks)</div><table><tr><th></th><th>NDVI</th><th>Proj NDVI</th>' + ('<th>Multi-Idx</th>' if multi_index_result else '') + '</tr><tr><td>NDVI</td><td>1.00</td><td>' + f'{iou_ndvi_emb:.2f}' + '</td>' + (f'<td>{iou_ndvi_multi:.2f}</td>' if multi_index_result else '') + '</tr><tr><td>Proj NDVI</td><td>' + f'{iou_ndvi_emb:.2f}' + '</td><td>1.00</td>' + (f'<td>{iou_emb_multi:.2f}</td>' if multi_index_result else '') + '</tr>' + (f'<tr><td>Multi-Idx</td><td>{iou_ndvi_multi:.2f}</td><td>{iou_emb_multi:.2f}</td><td>1.00</td></tr>' if multi_index_result else '') + '</table>'}
        {'<div class="section-title">Correlation Matrix (Continuous)</div><table><tr><th></th><th>NDVI</th><th>Proj NDVI</th>' + ('<th>Multi-Idx</th>' if multi_index_result else '') + '</tr><tr><td>NDVI</td><td>1.00</td><td>' + f'{corr_ndvi_emb:.2f}' + '</td>' + (f'<td>{corr_ndvi_multi:.2f}</td>' if multi_index_result else '') + '</tr><tr><td>Proj NDVI</td><td>' + f'{corr_ndvi_emb:.2f}' + '</td><td>1.00</td>' + (f'<td>{corr_emb_multi:.2f}</td>' if multi_index_result else '') + '</tr>' + (f'<tr><td>Multi-Idx</td><td>{corr_ndvi_multi:.2f}</td><td>{corr_emb_multi:.2f}</td><td>1.00</td></tr>' if multi_index_result else '') + '</table>'}
        <div class="section-title">R² (Embedding → Index)</div>
        <div class="metrics">
            Proj NDVI: <strong>{r2_emb:.3f}</strong>
            {(' | ' + ' | '.join([f'{name}: <strong>{r2val:.3f}</strong>' for name, r2val in multi_r2.items()])) if multi_r2 else ''}
        </div>
    </div>

    <div class="container">
        <div class="map-panel">
            <div class="panel-header">
                <select id="left-select"></select>
            </div>
            <div id="map-left" class="map"></div>
            <div class="panel-info" id="info-left"></div>
        </div>
        <div class="map-panel">
            <div class="panel-header">
                <select id="right-select"></select>
            </div>
            <div id="map-right" class="map"></div>
            <div class="panel-info" id="info-right"></div>
        </div>
    </div>

    <div id="controls">
        <label>Opacity</label>
        <input type="range" id="opacity-slider" min="0" max="100" value="80">
    </div>

    <script>
        var images = {json.dumps(images)};
        var layerInfo = {json.dumps(layer_info)};
        var bounds = {json.dumps(leaflet_bounds)};
        var center = {json.dumps(center)};

        var layers = [
            {{ key: 'emb_proj', name: 'Embedding Projection (continuous)' }},
            {{ key: 'ndvi_raw', name: 'Raw NDVI (continuous)' }},
            {{ key: 'multi_proj', name: 'Multi-Index Projection (continuous)' }},
            {{ key: 'emb_binary', name: 'Embedding Binary (unsmoothed)' }},
            {{ key: 'ndvi_binary', name: 'NDVI Binary (unsmoothed)' }},
            {{ key: 'multi_binary', name: 'Multi-Index Binary (K-means)' }}
        ];

        // Maps
        var mapLeft = L.map('map-left', {{zoomControl: false}}).setView(center, 12);
        var mapRight = L.map('map-right', {{zoomControl: false}}).setView(center, 12);
        L.control.zoom({{position: 'bottomleft'}}).addTo(mapLeft);

        // Basemaps
        var basemapUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}';
        L.tileLayer(basemapUrl, {{attribution: '© Esri'}}).addTo(mapLeft);
        L.tileLayer(basemapUrl, {{attribution: '© Esri'}}).addTo(mapRight);

        var leftOverlay = null;
        var rightOverlay = null;

        // Populate dropdowns
        function populateSelect(selectId, defaultIdx) {{
            var select = document.getElementById(selectId);
            layers.forEach(function(l, i) {{
                if (images[l.key]) {{
                    var opt = document.createElement('option');
                    opt.value = l.key;
                    opt.textContent = l.name;
                    if (i === defaultIdx) opt.selected = true;
                    select.appendChild(opt);
                }}
            }});
        }}
        populateSelect('left-select', 0);
        populateSelect('right-select', 1);

        // Update panel
        function updatePanel(side) {{
            var map = side === 'left' ? mapLeft : mapRight;
            var overlay = side === 'left' ? leftOverlay : rightOverlay;
            var selectId = side + '-select';
            var infoId = 'info-' + side;

            var key = document.getElementById(selectId).value;
            var info = layerInfo[key];

            if (overlay) map.removeLayer(overlay);

            if (images[key]) {{
                var opacity = document.getElementById('opacity-slider').value / 100;
                var newOverlay = L.imageOverlay('data:image/png;base64,' + images[key], bounds, {{
                    opacity: opacity
                }}).addTo(map);

                if (side === 'left') leftOverlay = newOverlay;
                else rightOverlay = newOverlay;
            }}

            // Update info
            var infoEl = document.getElementById(infoId);
            if (info) {{
                var html = '<div class="title">' + info.name + '</div>';
                if (info.type === 'continuous') {{
                    html += '<div class="stat"><span class="stat-label">Range:</span> ' + info.vmin + ' → ' + info.vmax + '</div>';
                    if (info.threshold) {{
                        html += '<div class="stat"><span class="stat-label">Threshold:</span> ' + info.threshold + '</div>';
                    }}
                    if (info.r2_per_index) {{
                        html += '<div class="stat"><span class="stat-label">R² per index:</span> ' + info.r2_per_index + '</div>';
                    }}
                }} else {{
                    html += '<div class="stat"><span class="stat-label">Veg Coverage:</span> ' + info.veg_coverage + '</div>';
                    if (info.threshold) {{
                        html += '<div class="stat"><span class="stat-label">Threshold:</span> ' + info.threshold + '</div>';
                    }}
                    if (info.method) {{
                        html += '<div class="stat"><span class="stat-label">Method:</span> ' + info.method + '</div>';
                    }}
                }}
                infoEl.innerHTML = html;
            }}
        }}

        // Event listeners
        document.getElementById('left-select').addEventListener('change', function() {{ updatePanel('left'); }});
        document.getElementById('right-select').addEventListener('change', function() {{ updatePanel('right'); }});
        document.getElementById('opacity-slider').addEventListener('input', function() {{
            var opacity = this.value / 100;
            if (leftOverlay) leftOverlay.setOpacity(opacity);
            if (rightOverlay) rightOverlay.setOpacity(opacity);
        }});

        // Sync maps
        var syncing = false;
        function syncMap(source, target) {{
            source.on('move', function() {{
                if (syncing) return;
                syncing = true;
                target.setView(source.getCenter(), source.getZoom(), {{animate: false}});
                syncing = false;
            }});
        }}
        syncMap(mapLeft, mapRight);
        syncMap(mapRight, mapLeft);

        // Initialize
        updatePanel('left');
        updatePanel('right');
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print_progress(f"Side-by-side raw viewer saved: {output_path}")
    return html


def create_comparison_visualization(
    embedding_result: dict,
    ndvi_result: dict,
    comparison: dict,
    output_path: str,
) -> str:
    """
    Create HTML visualization comparing both methods.

    Args:
        embedding_result: Embedding boundary result
        ndvi_result: NDVI boundary result
        comparison: Comparison metrics
        output_path: Output HTML path

    Returns:
        HTML content
    """
    meta = embedding_result.get('meta', ndvi_result.get('meta', {}))
    bounds = meta.get('bounds', None)

    if bounds:
        center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
        leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    else:
        center = [31.25, 34.8]
        leaflet_bounds = [[31.0, 34.5], [31.5, 35.1]]

    # Create map
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # Add basemap
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='&copy; Esri',
        name='Satellite',
    ).add_to(m)

    # Create agreement overlay
    agreement_rgb = create_agreement_overlay(
        embedding_result['boundary_mask'],
        ndvi_result['boundary_mask'],
        embedding_result['smoothed_mask'],
        ndvi_result['smoothed_mask'],
    )

    # Convert to base64
    img = Image.fromarray(agreement_rgb)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()

    # Add agreement overlay
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}",
        bounds=leaflet_bounds,
        opacity=0.6,
        name='Agreement Overlay',
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Metrics legend
    metrics_html = f'''
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 350px;
    ">
        <h4 style="margin: 0 0 10px 0;">Boundary Comparison Metrics</h4>

        <div style="margin-bottom: 8px;">
            <strong>Mask IoU:</strong> {comparison['iou']:.1%}
        </div>

        <div style="margin-bottom: 8px;">
            <strong>Boundary Agreement:</strong> {comparison['boundary_agreement']['mean_agreement']:.1%}
        </div>

        <table style="width: 100%; font-size: 11px; margin-top: 10px;">
            <tr style="background: #f0f0f0;">
                <th></th>
                <th>Embedding</th>
                <th>NDVI</th>
            </tr>
            <tr>
                <td>Threshold</td>
                <td>{comparison['thresholds']['embedding']:.4f}</td>
                <td>{comparison['thresholds']['ndvi']:.4f}</td>
            </tr>
            <tr>
                <td>Veg Coverage</td>
                <td>{comparison['vegetation_coverage']['embedding']:.1%}</td>
                <td>{comparison['vegetation_coverage']['ndvi']:.1%}</td>
            </tr>
            <tr>
                <td>Curvature Var</td>
                <td>{comparison['smoothness_embedding']['curvature_variance']:.4f}</td>
                <td>{comparison['smoothness_ndvi']['curvature_variance']:.4f}</td>
            </tr>
            <tr>
                <td>Fractal Dim</td>
                <td>{comparison['smoothness_embedding']['fractal_dimension']:.2f}</td>
                <td>{comparison['smoothness_ndvi']['fractal_dimension']:.2f}</td>
            </tr>
        </table>

        <div style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 8px;">
            <strong>Legend:</strong><br>
            <span style="color: green;">■</span> Both detect boundary<br>
            <span style="color: red;">■</span> Embedding only<br>
            <span style="color: blue;">■</span> NDVI only
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(metrics_html))

    html = m._repr_html_()

    with open(output_path, 'w') as f:
        f.write(html)

    return html


# =============================================================================
# CLI
# =============================================================================

def print_comparison_report(comparison: dict, embedding_result: dict, ndvi_result: dict):
    """Print formatted comparison report to console."""
    print("\n" + "=" * 70)
    print("BOUNDARY DETECTION COMPARISON")
    print("=" * 70)

    print(f"\n{'':30} {'Embedding':>15} {'NDVI Direct':>15}")
    print("-" * 70)
    print(f"{'Threshold (Otsu):':<30} {comparison['thresholds']['embedding']:>15.4f} {comparison['thresholds']['ndvi']:>15.4f}")
    print(f"{'Vegetation coverage:':<30} {comparison['vegetation_coverage']['embedding']*100:>14.1f}% {comparison['vegetation_coverage']['ndvi']*100:>14.1f}%")
    print(f"{'Boundary pixels:':<30} {np.sum(embedding_result['boundary_mask']):>15,} {np.sum(ndvi_result['boundary_mask']):>15,}")

    print("\n" + "-" * 70)
    print("COMPARISON METRICS")
    print("-" * 70)
    print(f"{'Vegetation Mask IoU:':<30} {comparison['iou']:>15.1%}")
    print(f"{'Boundary Agreement:':<30} {comparison['boundary_agreement']['mean_agreement']:>15.1%}")
    print(f"  └─ Embedding→NDVI: {comparison['boundary_agreement']['agreement_1_to_2']:.1%}")
    print(f"  └─ NDVI→Embedding: {comparison['boundary_agreement']['agreement_2_to_1']:.1%}")

    print("\n" + "-" * 70)
    print("SMOOTHNESS METRICS")
    print("-" * 70)
    print(f"{'':30} {'Embedding':>15} {'NDVI':>15} {'(lower=smoother)'}")
    print(f"{'Curvature Variance:':<30} {comparison['smoothness_embedding']['curvature_variance']:>15.4f} {comparison['smoothness_ndvi']['curvature_variance']:>15.4f}")
    print(f"{'Fractal Dimension:':<30} {comparison['smoothness_embedding']['fractal_dimension']:>15.2f} {comparison['smoothness_ndvi']['fractal_dimension']:>15.2f}")
    print(f"{'Sinuosity:':<30} {comparison['smoothness_embedding']['sinuosity']:>15.2f} {comparison['smoothness_ndvi']['sinuosity']:>15.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if comparison['summary']['embedding_smoother_by_pct'] > 10:
        print(f"Embedding method produces SMOOTHER boundaries ({comparison['summary']['embedding_smoother_by_pct']:.0f}% lower curvature variance)")
    elif comparison['summary']['embedding_smoother_by_pct'] < -10:
        print(f"NDVI method produces smoother boundaries ({-comparison['summary']['embedding_smoother_by_pct']:.0f}% lower curvature variance)")
    else:
        print("Both methods produce similarly smooth boundaries")

    if comparison['iou'] > 0.9:
        print("Vegetation masks are VERY SIMILAR (IoU > 90%)")
    elif comparison['iou'] > 0.8:
        print("Vegetation masks are SIMILAR (IoU > 80%)")
    else:
        print(f"Vegetation masks DIFFER significantly (IoU = {comparison['iou']:.1%})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Compare embedding vs NDVI boundary detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--embeddings', '-e',
        help='Path to Google Satellite Embeddings GeoTIFF (optional if cached)'
    )

    parser.add_argument(
        '--landsat', '-l',
        required=True,
        help='Path to Landsat GeoTIFF'
    )

    parser.add_argument(
        '--smooth-kernel',
        type=int,
        default=15,
        help='Morphological kernel radius (default: 15)'
    )

    parser.add_argument(
        '--smooth-iterations',
        type=int,
        default=5,
        help='Number of smoothing iterations (default: 5)'
    )

    parser.add_argument(
        '--tolerance',
        type=int,
        default=2,
        help='Boundary agreement tolerance in pixels (default: 2)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Recompute embedding boundary instead of using cached'
    )

    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip HTML visualization'
    )

    parser.add_argument(
        '--raw-visualization',
        action='store_true',
        help='Generate raw comparison visualization (before post-processing)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t open visualization in browser'
    )

    parser.add_argument(
        '-o', '--output-dir',
        default=OUTPUT_DIR,
        help='Output directory'
    )

    parser.add_argument(
        '--name', '-n',
        default=None,
        help='Name prefix for output files (e.g., "algeria" -> algeria_boundary_comparison.html)'
    )

    parser.add_argument(
        '--include-multi-index',
        action='store_true',
        help='Include multi-index (NDVI+SAVI+BSI) projection in comparison'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(BOUNDARY_DIR, exist_ok=True)

    # Helper to create filenames with optional prefix
    def make_filename(base_name):
        if args.name:
            return f"{args.name}_{base_name}"
        return base_name

    print("\n" + "=" * 70)
    print("BOUNDARY DETECTION COMPARISON")
    print("=" * 70)
    print(f"Landsat: {os.path.basename(args.landsat)}")
    print(f"Smooth kernel: {args.smooth_kernel}, iterations: {args.smooth_iterations}")

    # Step 1: NDVI boundary
    print("\n" + "-" * 70)
    print("[1/4] Computing NDVI boundary (baseline)")
    print("-" * 70)
    ndvi_result = compute_ndvi_boundary(
        args.landsat,
        smooth_kernel=args.smooth_kernel,
        smooth_iterations=args.smooth_iterations,
    )

    # Save NDVI boundary
    ndvi_npz_path = os.path.join(BOUNDARY_DIR, make_filename('boundary_ndvi.npz'))
    save_ndvi_boundary(ndvi_result, ndvi_npz_path)

    # Step 2: Embedding boundary
    print("\n" + "-" * 70)
    print("[2/4] Computing/loading embedding boundary")
    print("-" * 70)

    use_cached = not args.no_cache
    if args.embeddings:
        embedding_result = compute_embedding_boundary(
            args.embeddings,
            args.landsat,
            smooth_kernel=args.smooth_kernel,
            smooth_iterations=args.smooth_iterations,
            use_cached=use_cached,
        )
    elif use_cached and os.path.exists(os.path.join(BOUNDARY_DIR, 'boundary_embedding.npz')):
        embedding_result = compute_embedding_boundary(
            None, args.landsat, use_cached=True,
        )
    else:
        print("ERROR: No embeddings provided and no cached results found.")
        print("Please provide --embeddings path or run boundary_detector.py first.")
        sys.exit(1)

    # Save embedding boundary
    embedding_npz_path = os.path.join(BOUNDARY_DIR, make_filename('boundary_embedding.npz'))
    save_embedding_boundary(embedding_result, embedding_npz_path)

    # Create boundary visualization (red line on satellite)
    boundary_vis_path = os.path.join(args.output_dir, make_filename('boundary_visualization.html'))
    create_boundary_visualization(
        projection=embedding_result['projection'],
        smoothed_mask=embedding_result['smoothed_mask'],
        contours=embedding_result['contours'],
        meta=embedding_result['meta'],
        output_path=boundary_vis_path,
    )
    print(f"Boundary visualization saved: {boundary_vis_path}")

    # Step 3: Compare
    print("\n" + "-" * 70)
    print("[3/4] Computing comparison metrics")
    print("-" * 70)
    comparison = compare_boundaries(
        embedding_result,
        ndvi_result,
        tolerance_pixels=args.tolerance,
    )

    # Print report
    print_comparison_report(comparison, embedding_result, ndvi_result)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, make_filename('boundary_comparison_metrics.json'))
    with open(metrics_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Step 4: Visualization
    if not args.no_visualization:
        print("\n" + "-" * 70)
        print("[4/4] Creating visualization")
        print("-" * 70)

        vis_path = os.path.join(args.output_dir, make_filename('boundary_comparison.html'))
        create_comparison_visualization(
            embedding_result,
            ndvi_result,
            comparison,
            vis_path,
        )
        print(f"Visualization saved: {vis_path}")

        if not args.no_browser:
            webbrowser.open(f'file://{os.path.abspath(vis_path)}')

    # Step 5: Multi-index boundary (optional)
    multi_index_result = None
    if args.include_multi_index and args.embeddings:
        print("\n" + "-" * 70)
        print("[5/6] Computing multi-index boundary (NDVI+SAVI+BSI)")
        print("-" * 70)
        multi_index_result = run_multi_index_pipeline(
            embeddings_path=args.embeddings,
            landsat_path=args.landsat,
        )

    # Step 6: Raw visualization (before post-processing)
    if args.raw_visualization:
        print("\n" + "-" * 70)
        step_num = "6/6" if args.include_multi_index else "5/5"
        print(f"[{step_num}] Creating RAW comparison visualizations")
        print("-" * 70)

        # Single-map layer toggle version
        raw_vis_path = os.path.join(args.output_dir, make_filename('boundary_raw_comparison.html'))
        create_raw_comparison_visualization(
            embedding_result,
            ndvi_result,
            raw_vis_path,
        )

        # Side-by-side comparison version
        sidebyside_path = os.path.join(args.output_dir, make_filename('boundary_raw_sidebyside.html'))
        create_sidebyside_raw_viewer(
            embedding_result,
            ndvi_result,
            sidebyside_path,
            multi_index_result=multi_index_result,
        )

        if not args.no_browser:
            webbrowser.open(f'file://{os.path.abspath(sidebyside_path)}')


if __name__ == '__main__':
    main()
