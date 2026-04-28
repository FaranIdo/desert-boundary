#!/usr/bin/env python3
"""
Desert-Vegetation Boundary Detector.

Uses supervised linear projection to find the MAIN boundary between desert and
vegetated areas by projecting satellite embeddings onto an NDVI-aligned direction.

Technique: Supervised Linear Projection / Target-Aligned Dimensionality Reduction

Algorithm:
    1. Load embeddings (64D Google Satellite Embeddings) - downsample to 30m
    2. Compute NDVI from Landsat imagery (30m native)
    3. Fit linear regression: NDVI ~ embeddings
    4. Project embeddings onto regression weights (NDVI direction)
    5. Threshold using Otsu's method
    6. Heavy morphological smoothing to merge small regions
    7. Extract largest connected components for desert and vegetation
    8. Find the main boundary between the two regions
    9. Export as GeoJSON and visualization

Usage:
    python analysis/boundary_detector.py \
        --embeddings data/google_embedding_2022.tif \
        --landsat data/LC08_20221001_SAVI.tif

See docs/boundary_detection.md for detailed explanation.
"""

import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime
import time

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import Ridge
from scipy import ndimage
from skimage import filters, morphology, measure
import folium
from PIL import Image
from io import BytesIO
import base64
import geopandas as gpd

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Landsat 8/9 band indices (1-indexed in rasterio)
LANDSAT_RED_BAND = 4   # Band 4: Red (0.64-0.67 μm)
LANDSAT_NIR_BAND = 5   # Band 5: NIR (0.85-0.88 μm)


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
    print_progress(f"Embeddings resolution: ~{abs(meta['transform'][0]) * 111000:.1f}m")

    return embeddings, meta


def compute_ndvi_from_landsat(geotiff_path: str) -> tuple[np.ndarray, dict]:
    """
    Compute NDVI from Landsat 8/9 GeoTIFF.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        geotiff_path: Path to Landsat GeoTIFF

    Returns:
        ndvi: (H, W) float32 array with values in [-1, 1]
        meta: Dictionary with transform, crs, bounds, shape
    """
    print_progress(f"Computing NDVI from: {os.path.basename(geotiff_path)}")

    with rasterio.open(geotiff_path) as src:
        print_progress(f"Reading Red band (B{LANDSAT_RED_BAND})...")
        red = src.read(LANDSAT_RED_BAND).astype(np.float32)
        print_progress(f"Reading NIR band (B{LANDSAT_NIR_BAND})...")
        nir = src.read(LANDSAT_NIR_BAND).astype(np.float32)

        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'resolution': src.res,
        }

    print_progress("Calculating NDVI = (NIR - Red) / (NIR + Red)...")

    # Compute NDVI with epsilon to avoid division by zero
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)

    # Clip to valid range and handle invalid pixels
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = 0
    ndvi[(red == 0) & (nir == 0)] = 0  # No data pixels

    print_progress(f"NDVI shape: {ndvi.shape}")
    print_progress(f"NDVI resolution: ~{abs(meta['transform'][0]) * 111000:.1f}m")
    print_progress(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    return ndvi, meta


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
    print_progress(f"Processing {n_bands} bands...")

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
            resampling=Resampling.average,  # Average for downsampling
        )
        resampled[:, :, i] = destination

        if (i + 1) % 16 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_bands - i - 1)
            print_progress(f"  Resampled {i+1}/{n_bands} bands (ETA: {eta:.0f}s)", indent=2)

    print_progress(f"Resampling complete. New shape: {resampled.shape}")

    return resampled


# =============================================================================
# Supervised Linear Projection
# =============================================================================

def fit_ndvi_projection(
    embeddings: np.ndarray,
    ndvi: np.ndarray,
    subsample_ratio: float = 0.1,
    alpha: float = 1.0,
    random_state: int = 42
) -> tuple[np.ndarray, float, dict]:
    """
    Fit linear regression to find the NDVI direction in embedding space.

    Solves: NDVI ≈ embeddings @ weights + bias

    Args:
        embeddings: (H, W, 64) embedding array
        ndvi: (H, W) NDVI array (same grid as embeddings)
        subsample_ratio: Fraction of pixels for fitting
        alpha: Ridge regression regularization strength
        random_state: Random seed

    Returns:
        weights: (64,) weight vector defining NDVI direction
        bias: Scalar bias term
        stats: Dictionary with R², correlation, etc.
    """
    print_progress("Preparing data for regression...")

    h, w, n_dim = embeddings.shape

    # Flatten
    print_progress("Flattening arrays...")
    emb_flat = embeddings.reshape(-1, n_dim)
    ndvi_flat = ndvi.reshape(-1)

    # Find valid pixels (non-zero embeddings and valid NDVI)
    print_progress("Finding valid pixels...")
    valid_emb = np.any(emb_flat != 0, axis=1)
    valid_ndvi = (ndvi_flat != 0) & ~np.isnan(ndvi_flat)
    valid_mask = valid_emb & valid_ndvi

    X_valid = emb_flat[valid_mask]
    y_valid = ndvi_flat[valid_mask]

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

    # Fit Ridge regression
    print_progress("Fitting Ridge regression (alpha={:.1f})...".format(alpha))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_sample, y_sample)

    weights = model.coef_
    bias = model.intercept_

    # Compute R² on sample
    print_progress("Computing fit statistics...")
    y_pred = model.predict(X_sample)
    ss_res = np.sum((y_sample - y_pred) ** 2)
    ss_tot = np.sum((y_sample - y_sample.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Compute correlation
    correlation = np.corrcoef(y_sample, y_pred)[0, 1]

    print_progress(f"R² score: {r2:.4f}")
    print_progress(f"Correlation: {correlation:.4f}")
    print_progress(f"Weight norm: {np.linalg.norm(weights):.4f}")

    stats = {
        'r2': r2,
        'correlation': correlation,
        'weight_norm': np.linalg.norm(weights),
        'n_samples': n_samples,
        'n_valid': len(X_valid),
        'alpha': alpha,
    }

    return weights, bias, stats


def project_embeddings(
    embeddings: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0
) -> np.ndarray:
    """
    Project embeddings onto the NDVI direction.

    Args:
        embeddings: (H, W, 64) embedding array
        weights: (64,) weight vector
        bias: Scalar bias (optional)

    Returns:
        projection: (H, W) vegetation score array
    """
    print_progress("Computing dot product with weight vector...")

    # Dot product with weights
    projection = np.dot(embeddings, weights) + bias

    # Create validity mask
    print_progress("Applying validity mask...")
    valid_mask = np.any(embeddings != 0, axis=-1)
    projection[~valid_mask] = 0

    valid_proj = projection[valid_mask]
    print_progress(f"Projection range: [{valid_proj.min():.4f}, {valid_proj.max():.4f}]")
    print_progress(f"Projection mean: {valid_proj.mean():.4f}, std: {valid_proj.std():.4f}")

    return projection


# =============================================================================
# Thresholding and Main Boundary Extraction
# =============================================================================

def threshold_projection(
    projection: np.ndarray,
    threshold: float = None,
    method: str = 'otsu'
) -> tuple[np.ndarray, float]:
    """
    Threshold the projection to create binary desert/vegetation mask.

    Args:
        projection: (H, W) vegetation score array
        threshold: Manual threshold (if None, auto-compute)
        method: Auto-threshold method ('otsu' or 'mean')

    Returns:
        binary_mask: (H, W) bool array (True = vegetation)
        threshold: The threshold value used
    """
    print_progress("Computing threshold...")

    # Get valid pixels
    valid_mask = projection != 0
    valid_values = projection[valid_mask]

    if threshold is None:
        if method == 'otsu':
            # Normalize to [0, 1] for Otsu
            vmin, vmax = valid_values.min(), valid_values.max()
            normalized = (valid_values - vmin) / (vmax - vmin + 1e-10)
            otsu_normalized = filters.threshold_otsu(normalized)
            threshold = otsu_normalized * (vmax - vmin) + vmin
            print_progress(f"Otsu threshold: {threshold:.4f}")
        else:
            threshold = valid_values.mean()
            print_progress(f"Mean threshold: {threshold:.4f}")
    else:
        print_progress(f"Manual threshold: {threshold:.4f}")

    # Create binary mask
    print_progress("Creating binary mask...")
    binary_mask = (projection > threshold) & valid_mask

    # Statistics
    veg_pixels = binary_mask.sum()
    total_valid = valid_mask.sum()
    print_progress(f"Vegetation pixels: {veg_pixels:,} ({100*veg_pixels/total_valid:.1f}%)")
    print_progress(f"Desert pixels: {total_valid - veg_pixels:,} ({100*(total_valid-veg_pixels)/total_valid:.1f}%)")

    return binary_mask, threshold


def extract_main_boundary(
    binary_mask: np.ndarray,
    heavy_smooth_kernel_size: int = 15,
    heavy_smooth_iterations: int = 5,
    min_region_fraction: float = 0.05
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Extract the MAIN boundary between desert and vegetation.

    Uses heavy morphological smoothing to merge small regions,
    then keeps only the largest connected components.

    Args:
        binary_mask: (H, W) bool array (True = vegetation)
        heavy_smooth_kernel_size: Size of morphological kernel (larger = more smoothing)
        heavy_smooth_iterations: Number of smoothing iterations
        min_region_fraction: Minimum fraction of image for a region to be kept

    Returns:
        smoothed_mask: (H, W) bool array after heavy smoothing
        main_contours: List of contour arrays for the main boundary
        boundary_mask: (H, W) bool array with boundary pixels
    """
    print_progress("Extracting main boundary line...")

    h, w = binary_mask.shape
    total_pixels = h * w

    # Step 1: Heavy morphological smoothing
    print_progress(f"Applying heavy morphological smoothing (kernel={heavy_smooth_kernel_size}, iterations={heavy_smooth_iterations})...")

    mask_uint8 = binary_mask.astype(np.uint8)
    kernel = morphology.disk(heavy_smooth_kernel_size)

    for i in range(heavy_smooth_iterations):
        # Close (fill holes in vegetation)
        mask_uint8 = morphology.binary_closing(mask_uint8, kernel).astype(np.uint8)
        # Open (remove small vegetation patches)
        mask_uint8 = morphology.binary_opening(mask_uint8, kernel).astype(np.uint8)
        print_progress(f"  Smoothing iteration {i+1}/{heavy_smooth_iterations} complete", indent=2)

    smoothed_mask = mask_uint8.astype(bool)

    # Step 2: Label connected components for vegetation
    print_progress("Labeling connected components...")

    # Vegetation components
    veg_labels = measure.label(smoothed_mask, connectivity=2)
    veg_regions = measure.regionprops(veg_labels)

    # Desert components (inverse)
    desert_mask = ~smoothed_mask & (binary_mask | ~binary_mask)  # All non-veg valid pixels
    # Need to handle the valid area
    valid_mask = ndimage.binary_fill_holes(smoothed_mask | ~smoothed_mask)
    desert_in_valid = ~smoothed_mask
    desert_labels = measure.label(desert_in_valid, connectivity=2)
    desert_regions = measure.regionprops(desert_labels)

    print_progress(f"Found {len(veg_regions)} vegetation regions")
    print_progress(f"Found {len(desert_regions)} desert regions")

    # Step 3: Keep only largest regions
    print_progress("Keeping largest connected components...")

    min_pixels = int(total_pixels * min_region_fraction)
    print_progress(f"Minimum region size: {min_pixels:,} pixels ({min_region_fraction*100:.1f}% of image)")

    # Find largest vegetation region
    if veg_regions:
        veg_regions_sorted = sorted(veg_regions, key=lambda r: r.area, reverse=True)
        largest_veg = veg_regions_sorted[0]
        print_progress(f"Largest vegetation region: {largest_veg.area:,} pixels ({100*largest_veg.area/total_pixels:.1f}%)")

        # Create mask with only largest vegetation region
        final_veg_mask = (veg_labels == largest_veg.label)

        # Optionally include other large regions
        for region in veg_regions_sorted[1:]:
            if region.area >= min_pixels:
                print_progress(f"  Including vegetation region: {region.area:,} pixels", indent=2)
                final_veg_mask |= (veg_labels == region.label)
    else:
        final_veg_mask = smoothed_mask

    # Step 4: Additional smoothing on the final mask
    print_progress("Final boundary smoothing...")

    final_kernel = morphology.disk(5)
    final_mask = morphology.binary_closing(final_veg_mask, final_kernel)
    final_mask = morphology.binary_opening(final_mask, final_kernel)

    # Step 5: Extract contours
    print_progress("Extracting boundary contours...")

    contours_raw = measure.find_contours(final_mask.astype(np.uint8), level=0.5)

    # Sort by length and keep the longest ones
    contours_sorted = sorted(contours_raw, key=len, reverse=True)

    print_progress(f"Found {len(contours_sorted)} contours")
    if contours_sorted:
        print_progress(f"Longest contour: {len(contours_sorted[0]):,} points")
        if len(contours_sorted) > 1:
            print_progress(f"Second longest: {len(contours_sorted[1]):,} points")

    # Keep only significant contours (at least 1% of the longest)
    if contours_sorted:
        max_length = len(contours_sorted[0])
        min_length = max(100, int(max_length * 0.01))
        main_contours = [c for c in contours_sorted if len(c) >= min_length]
        print_progress(f"Keeping {len(main_contours)} main contours (>= {min_length} points)")
    else:
        main_contours = []

    # Step 6: Create boundary mask
    print_progress("Creating boundary mask...")

    sobel_x = ndimage.sobel(final_mask.astype(np.uint8), axis=1)
    sobel_y = ndimage.sobel(final_mask.astype(np.uint8), axis=0)
    boundary_mask = np.hypot(sobel_x, sobel_y) > 0

    print_progress(f"Boundary pixels: {boundary_mask.sum():,}")

    return final_mask, main_contours, boundary_mask


# =============================================================================
# GeoJSON Export
# =============================================================================

def contours_to_geojson(
    contours: list,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326"
) -> dict:
    """
    Convert contours to GeoJSON format.

    Args:
        contours: List of (N, 2) arrays with (row, col) coordinates
        transform: Rasterio affine transform
        crs: Coordinate reference system

    Returns:
        GeoJSON FeatureCollection dictionary
    """
    features = []

    for i, contour in enumerate(contours):
        # Convert pixel coordinates to geographic coordinates
        # contour is in (row, col) format, transform expects (col, row)
        coords = []
        for row, col in contour:
            x, y = transform * (col, row)
            coords.append([x, y])

        # Close the polygon if not already closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "length": len(contour),
                "type": "desert_vegetation_boundary",
                "rank": i + 1  # 1 = longest
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features
    }

    return geojson


# =============================================================================
# Visualization
# =============================================================================

def create_boundary_visualization(
    projection: np.ndarray,
    smoothed_mask: np.ndarray,
    contours: list,
    meta: dict,
    output_path: str = None,
    zoom: int = 12,
    reference_shapefiles: list = None
) -> str:
    """
    Create Folium map with boundary visualization.

    Args:
        projection: (H, W) vegetation score array
        smoothed_mask: (H, W) bool mask after smoothing
        contours: List of contour arrays
        meta: Dictionary with bounds, transform, crs
        output_path: Path to save HTML
        zoom: Initial zoom level
        reference_shapefiles: Optional list of (path, name, color) tuples for reference lines

    Returns:
        HTML content string
    """
    print_progress("Creating Folium map...")

    bounds = meta['bounds']
    center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
    leaflet_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    # Create map
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    # Add satellite basemap
    print_progress("Adding satellite basemap...")
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='&copy; Esri',
        name='Satellite',
    ).add_to(m)

    # Add smoothed mask overlay
    print_progress("Creating mask overlay...")

    # Create RGBA image for smoothed mask
    cmap = np.zeros((*smoothed_mask.shape, 4), dtype=np.uint8)
    # Green for vegetation, transparent for desert
    cmap[smoothed_mask, 1] = 150  # Green
    cmap[smoothed_mask, 3] = 120  # Alpha
    cmap[~smoothed_mask, 0] = 150  # Red for desert
    cmap[~smoothed_mask, 3] = 80   # Lower alpha for desert

    # Convert to base64
    img = Image.fromarray(cmap)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()

    # Add mask overlay
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}",
        bounds=leaflet_bounds,
        opacity=0.5,
        name='Vegetation Mask (smoothed)',
    ).add_to(m)

    # Add boundary contours
    print_progress(f"Adding {len(contours)} boundary contours...")

    transform = meta['transform']
    boundary_group = folium.FeatureGroup(name='Main Boundary')

    for idx, contour in enumerate(contours):
        # Convert to lat/lon
        coords = []
        for row, col in contour:
            x, y = transform * (col, row)
            coords.append([y, x])  # Folium uses [lat, lon]

        # Color: red for main, orange for secondary
        color = '#FF0000' if idx == 0 else '#FF8800'
        weight = 4 if idx == 0 else 2

        # Add polyline
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.9,
            popup=f"Boundary #{idx+1} ({len(contour):,} points)"
        ).add_to(boundary_group)

    boundary_group.add_to(m)

    # Add reference shapefiles if provided
    if reference_shapefiles:
        for shapefile_path, layer_name, color in reference_shapefiles:
            if os.path.exists(shapefile_path):
                print_progress(f"Adding reference boundary: {layer_name}")
                try:
                    gdf = gpd.read_file(shapefile_path)
                    reference_group = folium.FeatureGroup(name=layer_name, show=True)

                    # Style for reference line
                    style = {
                        'color': color,
                        'weight': 3,
                        'opacity': 0.8
                    }

                    folium.GeoJson(
                        gdf,
                        style_function=lambda x, c=color: {'color': c, 'weight': 3, 'opacity': 0.8},
                        tooltip=layer_name
                    ).add_to(reference_group)

                    reference_group.add_to(m)
                    print_progress(f"  Added in {color}")
                except Exception as e:
                    print_progress(f"Warning: Could not load {shapefile_path}: {e}")

    # Add layer control
    folium.LayerControl().add_to(m)

    # Build dynamic legend based on reference shapefiles
    reference_legend_items = ""
    if reference_shapefiles:
        for _, layer_name, color in reference_shapefiles:
            reference_legend_items += f'''
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 20px; height: 3px; background: {color}; margin-right: 8px;"></span>
            {layer_name}
        </div>'''

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
        font-size: 13px;
    ">
        <h4 style="margin: 0 0 10px 0;">Desert-Vegetation Boundary</h4>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 20px; height: 4px; background: #FF0000; margin-right: 8px;"></span>
            Detected Main Boundary
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 20px; height: 2px; background: #FF8800; margin-right: 8px;"></span>
            Detected Secondary Boundaries
        </div>{reference_legend_items}
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 20px; height: 12px; background: rgba(0,150,0,0.5); margin-right: 8px;"></span>
            Vegetation
        </div>
        <div>
            <span style="display: inline-block; width: 20px; height: 12px; background: rgba(150,0,0,0.3); margin-right: 8px;"></span>
            Desert
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    html = m._repr_html_()

    if output_path:
        print_progress(f"Saving to: {output_path}")
        with open(output_path, 'w') as f:
            f.write(html)

    return html


# =============================================================================
# Main Pipeline
# =============================================================================

def detect_desert_vegetation_boundary(
    embeddings_path: str,
    landsat_path: str,
    threshold: float = None,
    heavy_smooth_kernel: int = 15,
    heavy_smooth_iterations: int = 5,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_geojson: bool = True,
    save_visualization: bool = True,
    open_browser: bool = True,
    reference_shapefiles: list = None
) -> dict:
    """
    Full pipeline for detecting the MAIN desert-vegetation boundary.

    Args:
        embeddings_path: Path to Google Satellite Embeddings GeoTIFF
        landsat_path: Path to Landsat GeoTIFF
        threshold: Manual threshold (None for auto Otsu)
        heavy_smooth_kernel: Kernel size for heavy smoothing
        heavy_smooth_iterations: Number of heavy smoothing iterations
        output_dir: Output directory
        save_geojson: Whether to save GeoJSON
        save_visualization: Whether to save HTML visualization
        open_browser: Whether to open visualization in browser
        reference_shapefiles: Optional list of (path, name, color) tuples for reference lines

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)

    total_steps = 7
    start_time = time.time()

    print("\n" + "=" * 60)
    print("DESERT-VEGETATION MAIN BOUNDARY DETECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load NDVI (target resolution)
    print_step(1, total_steps, "Loading Landsat and computing NDVI")
    ndvi, ndvi_meta = compute_ndvi_from_landsat(landsat_path)

    # Step 2: Load embeddings
    print_step(2, total_steps, "Loading embeddings")
    embeddings_raw, emb_meta = load_embeddings(embeddings_path)

    # Step 3: Downsample embeddings to NDVI resolution (30m)
    print_step(3, total_steps, "Downsampling embeddings to 30m resolution")
    if embeddings_raw.shape[:2] != ndvi.shape:
        embeddings = resample_embeddings_to_target(embeddings_raw, emb_meta, ndvi_meta)
        # Update meta to use NDVI's transform
        working_meta = ndvi_meta.copy()
    else:
        embeddings = embeddings_raw
        working_meta = emb_meta
    print_progress(f"Working resolution: {embeddings.shape[:2]}")

    # Step 4: Fit supervised projection
    print_step(4, total_steps, "Fitting supervised linear projection (NDVI direction)")
    weights, bias, fit_stats = fit_ndvi_projection(embeddings, ndvi)

    # Step 5: Project embeddings
    print_step(5, total_steps, "Projecting embeddings onto NDVI direction")
    projection = project_embeddings(embeddings, weights, bias)

    # Step 6: Threshold
    print_step(6, total_steps, "Thresholding projection")
    binary_mask, used_threshold = threshold_projection(projection, threshold)

    # Step 7: Extract main boundary
    print_step(7, total_steps, "Extracting main boundary (heavy smoothing + largest components)")
    smoothed_mask, main_contours, boundary_mask = extract_main_boundary(
        binary_mask,
        heavy_smooth_kernel_size=heavy_smooth_kernel,
        heavy_smooth_iterations=heavy_smooth_iterations
    )

    # Build results
    elapsed = time.time() - start_time

    results = {
        'weights': weights,
        'bias': bias,
        'fit_stats': fit_stats,
        'projection': projection,
        'binary_mask': binary_mask,
        'smoothed_mask': smoothed_mask,
        'threshold': used_threshold,
        'boundary_mask': boundary_mask,
        'contours': main_contours,
        'meta': working_meta,
        'metadata': {
            'embeddings_file': os.path.basename(embeddings_path),
            'landsat_file': os.path.basename(landsat_path),
            'timestamp': datetime.now().isoformat(),
            'heavy_smooth_kernel': heavy_smooth_kernel,
            'heavy_smooth_iterations': heavy_smooth_iterations,
            'elapsed_seconds': elapsed,
        }
    }

    # Save outputs
    print("\n" + "-" * 50)
    print("SAVING OUTPUTS")
    print("-" * 50)

    # Save NPZ file (for unified viewer integration)
    boundary_dir = os.path.join(PROJECT_DIR, 'data', 'boundary')
    os.makedirs(boundary_dir, exist_ok=True)

    # Create boundary RGB visualization (green=vegetation, brown=desert, red=boundary)
    boundary_rgb = np.zeros((*smoothed_mask.shape, 3), dtype=np.uint8)
    boundary_rgb[smoothed_mask, 1] = 120  # Green for vegetation
    boundary_rgb[~smoothed_mask, 0] = 120  # Red component for desert
    boundary_rgb[~smoothed_mask, 1] = 80   # Some green for desert (brownish)
    boundary_rgb[boundary_mask, 0] = 255   # Red boundary line
    boundary_rgb[boundary_mask, 1] = 0
    boundary_rgb[boundary_mask, 2] = 0

    npz_filename = 'boundary_embedding.npz'
    npz_path = os.path.join(boundary_dir, npz_filename)

    # Convert contours to saveable format (list of arrays)
    contours_list = [c.astype(np.float32) for c in main_contours]

    np.savez_compressed(
        npz_path,
        smoothed_mask=smoothed_mask.astype(np.uint8),
        boundary_mask=boundary_mask.astype(np.uint8),
        boundary_rgb=boundary_rgb,
        projection=projection.astype(np.float32),
        threshold=np.array([used_threshold]),
        weights=weights,
        bias=np.array([bias]),
        n_contours=np.array([len(main_contours)]),
        fit_stats=fit_stats,
        metadata={
            'embeddings_file': os.path.basename(embeddings_path),
            'landsat_file': os.path.basename(landsat_path),
            'timestamp': datetime.now().isoformat(),
            'heavy_smooth_kernel': heavy_smooth_kernel,
            'heavy_smooth_iterations': heavy_smooth_iterations,
            'r2': fit_stats['r2'],
            'correlation': fit_stats['correlation'],
        }
    )
    print_progress(f"NPZ saved: {npz_path}")
    results['npz_path'] = npz_path

    # Save GeoJSON
    if save_geojson and main_contours:
        geojson = contours_to_geojson(main_contours, working_meta['transform'])
        geojson_path = os.path.join(output_dir, 'desert_vegetation_boundary.geojson')
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print_progress(f"GeoJSON saved: {geojson_path}")
        results['geojson_path'] = geojson_path

    # Save visualization
    if save_visualization:
        vis_path = os.path.join(output_dir, 'boundary_visualization.html')
        create_boundary_visualization(
            projection, smoothed_mask, main_contours, working_meta,
            output_path=vis_path,
            reference_shapefiles=reference_shapefiles
        )
        results['visualization_path'] = vis_path

        if open_browser:
            print_progress("Opening in browser...")
            webbrowser.open(f'file://{os.path.abspath(vis_path)}')

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Main contours found: {len(main_contours)}")
    if main_contours:
        print(f"Longest boundary: {len(main_contours[0]):,} points")
    print("=" * 60)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Detect the MAIN desert-vegetation boundary using supervised linear projection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python boundary_detector.py \\
      --embeddings data/google_embedding_2022.tif \\
      --landsat data/LC08_20221001_SAVI.tif

  python boundary_detector.py \\
      --embeddings data/google_embedding_2022.tif \\
      --landsat data/LC08_20221001_SAVI.tif \\
      --smooth-kernel 20 --smooth-iterations 7
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
        help='Path to Landsat 8/9 GeoTIFF (for NDVI calculation)'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Manual threshold (default: auto Otsu)'
    )

    parser.add_argument(
        '--smooth-kernel',
        type=int,
        default=15,
        help='Kernel size for heavy morphological smoothing (default: 15)'
    )

    parser.add_argument(
        '--smooth-iterations',
        type=int,
        default=5,
        help='Number of heavy smoothing iterations (default: 5)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open visualization in browser'
    )

    parser.add_argument(
        '--no-geojson',
        action='store_true',
        help='Do not save GeoJSON output'
    )

    parser.add_argument(
        '--reference-shapefiles',
        nargs='*',
        default=None,
        help='Reference shapefiles in format: path:name:color (e.g., edge/phyto-line.shp:Phytogeographic:#0000FF)'
    )

    args = parser.parse_args()

    # Parse reference shapefiles
    reference_shapefiles = None
    if args.reference_shapefiles:
        reference_shapefiles = []
        default_colors = ['#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF']
        for idx, ref in enumerate(args.reference_shapefiles):
            parts = ref.split(':')
            path = parts[0]
            name = parts[1] if len(parts) > 1 else os.path.basename(path).replace('.shp', '')
            color = parts[2] if len(parts) > 2 else default_colors[idx % len(default_colors)]
            reference_shapefiles.append((path, name, color))

    # Validate inputs
    if not os.path.exists(args.embeddings):
        parser.error(f"Embeddings file not found: {args.embeddings}")
    if not os.path.exists(args.landsat):
        parser.error(f"Landsat file not found: {args.landsat}")

    # Run pipeline
    results = detect_desert_vegetation_boundary(
        embeddings_path=args.embeddings,
        landsat_path=args.landsat,
        threshold=args.threshold,
        heavy_smooth_kernel=args.smooth_kernel,
        heavy_smooth_iterations=args.smooth_iterations,
        output_dir=args.output_dir,
        save_geojson=not args.no_geojson,
        save_visualization=True,
        open_browser=not args.no_browser,
        reference_shapefiles=reference_shapefiles
    )

    return results


if __name__ == '__main__':
    main()
