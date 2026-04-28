#!/usr/bin/env python3
"""
GMM-based boundary detection on satellite embeddings.

Uses Gaussian Mixture Model to find probabilistic desert/vegetation boundaries
without requiring labels. Outputs probability scores per pixel.

Usage:
    python analysis/gmm_boundary.py --embeddings data/google_embedding_2022.tif
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import webbrowser


def load_embeddings(embeddings_path, downsample=3):
    """Load embeddings with optional downsampling and return array + metadata."""
    print(f"Loading embeddings from {embeddings_path}...")
    with rasterio.open(embeddings_path) as src:
        # Read with downsampling to reduce memory
        if downsample > 1:
            from rasterio.enums import Resampling
            out_shape = (
                src.count,
                src.height // downsample,
                src.width // downsample
            )
            embeddings = src.read(out_shape=out_shape, resampling=Resampling.average)
            # Adjust transform for downsampled resolution
            transform = src.transform * src.transform.scale(downsample, downsample)
            print(f"  Downsampled {downsample}x: {src.height}x{src.width} -> {out_shape[1]}x{out_shape[2]}")
        else:
            embeddings = src.read()
            transform = src.transform

        crs = src.crs
        bounds = src.bounds

    # Reshape to (H*W, 64)
    n_bands, height, width = embeddings.shape
    embeddings_flat = embeddings.reshape(n_bands, -1).T  # (H*W, 64)

    print(f"  Shape: {height}x{width} pixels, {n_bands} dimensions")
    print(f"  Total pixels: {height * width:,}")

    return embeddings_flat, (height, width), transform, crs, bounds


def fit_gmm(embeddings, n_components=2, random_state=42):
    """Fit GMM to embeddings and return probabilities."""
    print(f"\nFitting GMM with {n_components} components...")

    # Handle NaN/Inf values
    valid_mask = np.all(np.isfinite(embeddings), axis=1)
    valid_embeddings = embeddings[valid_mask]

    print(f"  Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,}")

    # Subsample for faster fitting if large
    n_samples = len(valid_embeddings)
    max_fit_samples = 100_000  # Reduced for memory
    if n_samples > max_fit_samples:
        sample_idx = np.random.RandomState(random_state).choice(
            n_samples, size=max_fit_samples, replace=False
        )
        fit_data = valid_embeddings[sample_idx]
        print(f"  Subsampled to {max_fit_samples:,} for fitting")
    else:
        fit_data = valid_embeddings

    # Standardize for better GMM performance
    scaler = StandardScaler()
    fit_data_scaled = scaler.fit_transform(fit_data)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=3,
        max_iter=200,
        verbose=1
    )
    gmm.fit(fit_data_scaled)

    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")

    # Predict on all valid pixels in batches to save memory
    print("  Predicting probabilities (in batches)...")
    batch_size = 50_000
    n_valid = len(valid_embeddings)
    probabilities = np.zeros((n_valid, n_components))
    labels = np.zeros(n_valid, dtype=int)

    for i in range(0, n_valid, batch_size):
        end = min(i + batch_size, n_valid)
        batch_scaled = scaler.transform(valid_embeddings[i:end])
        probabilities[i:end] = gmm.predict_proba(batch_scaled)
        labels[i:end] = gmm.predict(batch_scaled)
        if (i // batch_size) % 10 == 0:
            print(f"    Processed {end:,} / {n_valid:,} pixels")

    # Create full arrays with NaN for invalid pixels
    full_proba = np.full((len(embeddings), n_components), np.nan)
    full_proba[valid_mask] = probabilities

    full_labels = np.full(len(embeddings), -1)
    full_labels[valid_mask] = labels

    return gmm, scaler, full_proba, full_labels, valid_mask


def identify_vegetation_component(probabilities, labels, embeddings, shape, landsat_path=None):
    """Identify which GMM component corresponds to vegetation."""

    # If landsat provided, use NDVI correlation
    if landsat_path and Path(landsat_path).exists():
        print("\nUsing NDVI to identify vegetation component...")
        with rasterio.open(landsat_path) as src:
            bands = src.read()

        # Compute NDVI (assuming bands 4=Red, 5=NIR for Landsat 8/9)
        if bands.shape[0] >= 5:
            red = bands[3].astype(float)
            nir = bands[4].astype(float)
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi_flat = ndvi.flatten()

            # Resample to match embedding resolution if needed
            height, width = shape
            if len(ndvi_flat) != height * width:
                from scipy.ndimage import zoom
                zoom_factor = (height / ndvi.shape[0], width / ndvi.shape[1])
                ndvi_resampled = zoom(ndvi, zoom_factor, order=1)
                ndvi_flat = ndvi_resampled.flatten()

            # Check correlation with each component
            valid = ~np.isnan(probabilities[:, 0]) & ~np.isnan(ndvi_flat)
            n_components = probabilities.shape[1]
            correlations = []
            for i in range(n_components):
                corr = np.corrcoef(probabilities[valid, i], ndvi_flat[valid])[0, 1]
                correlations.append(corr)
                print(f"  Component {i} correlation with NDVI: {corr:.3f}")

            vegetation_component = np.argmax(correlations)
            print(f"  Vegetation component: {vegetation_component}")
            return vegetation_component

    # Fallback: assume component with fewer pixels is vegetation (desert dominates)
    print("\nIdentifying vegetation component by size (assuming desert dominates)...")
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    vegetation_component = unique[np.argmin(counts)]
    print(f"  Component sizes: {dict(zip(unique, counts))}")
    print(f"  Vegetation component (smaller): {vegetation_component}")
    return vegetation_component


def create_visualization(proba_map, shape, bounds, crs, output_path,
                        vegetation_component=0, threshold=0.5):
    """Create Folium HTML visualization."""
    print(f"\nCreating visualization...")

    height, width = shape
    vegetation_proba = proba_map[:, vegetation_component].reshape(height, width)

    # Handle NaN
    vegetation_proba = np.nan_to_num(vegetation_proba, nan=0.0)

    # Create RGB visualization (green = vegetation, yellow = desert)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Color gradient: brown (desert) -> green (vegetation)
    # Desert (low prob): RGB(194, 178, 128)
    # Vegetation (high prob): RGB(34, 139, 34)
    p = vegetation_proba
    rgb[:, :, 0] = (194 * (1 - p) + 34 * p).astype(np.uint8)   # R
    rgb[:, :, 1] = (178 * (1 - p) + 139 * p).astype(np.uint8)  # G
    rgb[:, :, 2] = (128 * (1 - p) + 34 * p).astype(np.uint8)   # B

    # Create boundary overlay (where probability ~ 0.5)
    boundary_zone = np.abs(vegetation_proba - threshold) < 0.1
    boundary_rgb = rgb.copy()
    boundary_rgb[boundary_zone] = [255, 0, 0]  # Red for boundary zone

    # Calculate center
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='Esri.WorldImagery',
        attr='Esri'
    )

    # Add probability overlay
    from PIL import Image
    import io
    import base64

    img = Image.fromarray(rgb)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        name='GMM Probability (Green=Vegetation)',
    ).add_to(m)

    # Add boundary zone overlay
    img_boundary = Image.fromarray(boundary_rgb)
    img_boundary_bytes = io.BytesIO()
    img_boundary.save(img_boundary_bytes, format='PNG')
    img_boundary_base64 = base64.b64encode(img_boundary_bytes.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_boundary_base64}',
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        name='Boundary Zone (Red = P~0.5)',
        show=False,
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">GMM Vegetation Probability</h4>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: rgb(34,139,34); margin-right: 10px;"></div>
            <span>High (Vegetation)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: rgb(114,158,81); margin-right: 10px;"></div>
            <span>Medium (Transition)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: rgb(194,178,128); margin-right: 10px;"></div>
            <span>Low (Desert)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: red; margin-right: 10px;"></div>
            <span>Boundary Zone (P~0.5)</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    m.save(output_path)
    print(f"  Saved to {output_path}")

    return vegetation_proba


def save_results(proba_map, labels, shape, transform, crs, vegetation_component,
                output_dir, valid_mask):
    """Save results to NPZ and GeoTIFF."""
    height, width = shape

    # Reshape to image
    vegetation_proba = proba_map[:, vegetation_component].reshape(height, width)
    labels_img = labels.reshape(height, width)

    # Save NPZ
    npz_path = output_dir / 'gmm_boundary.npz'
    np.savez_compressed(
        npz_path,
        vegetation_probability=vegetation_proba,
        labels=labels_img,
        probabilities=proba_map.reshape(height, width, -1),
        vegetation_component=vegetation_component,
        valid_mask=valid_mask.reshape(height, width),
    )
    print(f"  Saved NPZ to {npz_path}")

    # Save GeoTIFF of probability
    tif_path = output_dir / 'gmm_vegetation_probability.tif'
    with rasterio.open(
        tif_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(vegetation_proba.astype(np.float32), 1)
    print(f"  Saved GeoTIFF to {tif_path}")

    return npz_path, tif_path


def print_statistics(proba_map, labels, vegetation_component, shape):
    """Print summary statistics."""
    height, width = shape
    vegetation_proba = proba_map[:, vegetation_component]

    valid = ~np.isnan(vegetation_proba)

    print("\n" + "="*60)
    print("GMM BOUNDARY DETECTION RESULTS")
    print("="*60)

    print(f"\nProbability Statistics:")
    print(f"  Mean P(vegetation):   {np.nanmean(vegetation_proba):.3f}")
    print(f"  Std P(vegetation):    {np.nanstd(vegetation_proba):.3f}")
    print(f"  Min:                  {np.nanmin(vegetation_proba):.3f}")
    print(f"  Max:                  {np.nanmax(vegetation_proba):.3f}")

    # Coverage at different thresholds
    print(f"\nVegetation Coverage at Different Thresholds:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        coverage = np.sum(vegetation_proba[valid] > thresh) / valid.sum() * 100
        print(f"  P > {thresh}: {coverage:.1f}%")

    # Boundary zone (where probability is uncertain)
    boundary_pixels = np.sum((vegetation_proba[valid] > 0.4) & (vegetation_proba[valid] < 0.6))
    boundary_pct = boundary_pixels / valid.sum() * 100
    print(f"\nBoundary Zone (0.4 < P < 0.6): {boundary_pixels:,} pixels ({boundary_pct:.1f}%)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='GMM boundary detection on embeddings')
    parser.add_argument('--embeddings', '-e', type=str,
                       default='data/google_embedding_2022.tif',
                       help='Path to embeddings GeoTIFF')
    parser.add_argument('--landsat', '-l', type=str, default=None,
                       help='Path to Landsat GeoTIFF (for NDVI-based component identification)')
    parser.add_argument('--n-components', '-n', type=int, default=2,
                       help='Number of GMM components (default: 2)')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--downsample', '-d', type=int, default=3,
                       help='Downsample factor (default: 3 for memory efficiency)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load embeddings
    embeddings, shape, transform, crs, bounds = load_embeddings(
        args.embeddings, downsample=args.downsample
    )

    # Fit GMM
    gmm, scaler, proba_map, labels, valid_mask = fit_gmm(
        embeddings, n_components=args.n_components
    )

    # Identify vegetation component
    vegetation_component = identify_vegetation_component(
        proba_map, labels, embeddings, shape, args.landsat
    )

    # Print statistics
    print_statistics(proba_map, labels, vegetation_component, shape)

    # Save results
    data_dir = Path('data/boundary')
    data_dir.mkdir(parents=True, exist_ok=True)
    save_results(proba_map, labels, shape, transform, crs, vegetation_component,
                data_dir, valid_mask)

    # Create visualization
    html_path = output_dir / 'gmm_boundary.html'
    create_visualization(proba_map, shape, bounds, crs, html_path, vegetation_component)

    if not args.no_browser:
        webbrowser.open(f'file://{html_path.absolute()}')


if __name__ == '__main__':
    main()
