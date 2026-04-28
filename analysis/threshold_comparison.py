"""Compare different threshold factors for boundary detection."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
import geopandas as gpd
from skimage import measure
from skimage.morphology import disk, binary_closing, binary_opening
from sklearn.linear_model import Ridge


def load_rgb_image(tif_path, bands=[4, 3, 2], percentile=(2, 98)):
    """Load and normalize RGB from GeoTIFF."""
    with rasterio.open(tif_path) as src:
        rgb = np.stack([src.read(b) for b in bands], axis=-1).astype(float)
        transform = src.transform
        crs = src.crs
    for i in range(3):
        valid = rgb[:, :, i][rgb[:, :, i] > 0]
        if len(valid) > 0:
            p_low, p_high = np.percentile(valid, percentile)
            rgb[:, :, i] = np.clip((rgb[:, :, i] - p_low) / (p_high - p_low), 0, 1)
    return rgb, transform, crs


def compute_boundary_with_factor(embed_path, ndvi_path, threshold_factor=1.0):
    """Compute boundary with adjustable threshold factor."""
    from skimage.transform import resize as sk_resize

    # Load embeddings
    with rasterio.open(embed_path) as src:
        embeddings = src.read()  # (64, H, W)
    embeddings = np.moveaxis(embeddings, 0, -1)  # (H, W, 64)

    # Load NDVI
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)

    # Resize embeddings to match NDVI (NDVI is at 30m, embeddings at 10m)
    if embeddings.shape[:2] != ndvi.shape:
        embeddings = sk_resize(embeddings, (ndvi.shape[0], ndvi.shape[1], embeddings.shape[2]),
                                order=1, preserve_range=True)

    h, w, d = embeddings.shape

    # Flatten
    flat_embed = embeddings.reshape(-1, d)
    flat_ndvi = ndvi.reshape(-1)

    # Valid mask
    valid = np.isfinite(flat_embed).all(axis=1) & np.isfinite(flat_ndvi) & (flat_ndvi != 0)

    # Subsample for training
    valid_idx = np.where(valid)[0]
    n_samples = min(len(valid_idx) // 10, 100000)
    sample_idx = np.random.choice(valid_idx, size=n_samples, replace=False)

    # Fit Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(flat_embed[sample_idx], flat_ndvi[sample_idx])

    # Project all valid pixels
    projection = np.full(h * w, np.nan)
    projection[valid] = flat_embed[valid] @ ridge.coef_ + ridge.intercept_
    projection = projection.reshape(h, w)

    # Otsu threshold
    valid_proj = projection[np.isfinite(projection)]
    from skimage.filters import threshold_otsu
    otsu_thresh = threshold_otsu(valid_proj)

    # Apply factor
    adjusted_thresh = otsu_thresh * threshold_factor

    # Binary mask
    mask = projection > adjusted_thresh
    mask = np.nan_to_num(mask, nan=0).astype(bool)

    # Morphological smoothing
    kernel = disk(15)
    smoothed = binary_closing(mask, kernel)
    smoothed = binary_opening(smoothed, kernel)

    # Additional iterations
    for _ in range(4):
        smoothed = binary_closing(smoothed, kernel)
        smoothed = binary_opening(smoothed, kernel)

    coverage = smoothed.mean() * 100

    return smoothed, otsu_thresh, adjusted_thresh, coverage


def geo_to_pixel(geom, transform):
    """Convert geometry to pixel coordinates."""
    from rasterio.transform import rowcol
    coords = []
    if geom.geom_type == 'LineString':
        for x, y in geom.coords:
            row, col = rowcol(transform, x, y)
            coords.append((col, row))
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            for x, y in line.coords:
                row, col = rowcol(transform, x, y)
                coords.append((col, row))
    return coords


def plot_threshold_comparison(output_path='outputs/threshold_comparison.png'):
    """Create 4-panel comparison of threshold factors."""

    embed_path = 'data/beer_sheva/google_embedding_beer_sheva_2022.tif'
    ndvi_path = 'data/beer_sheva/LC08_20221001_SAVI.tif'
    edge_path = 'data/beer_sheva/edge/phyto-line.shp'

    # Load reference edge
    edge_gdf = gpd.read_file(edge_path)

    # Load RGB
    rgb, transform, crs = load_rgb_image(ndvi_path)

    # Reproject edge if needed
    if edge_gdf.crs != crs:
        edge_gdf = edge_gdf.to_crs(crs)

    # Threshold factors to compare
    factors = [1.0, 0.95, 0.90, 0.85]
    titles = ['(a) Original (Otsu)', '(b) Factor 0.95', '(c) Factor 0.90', '(d) Factor 0.85']

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    np.random.seed(42)  # For reproducibility

    for ax, factor, title in zip(axes, factors, titles):
        print(f"Computing with factor={factor}...")
        mask, otsu_thresh, adj_thresh, coverage = compute_boundary_with_factor(
            embed_path, ndvi_path, threshold_factor=factor
        )

        # Resize mask to match RGB if needed
        if mask.shape != rgb.shape[:2]:
            from skimage.transform import resize
            mask = resize(mask.astype(float), rgb.shape[:2], order=0) > 0.5

        # Plot RGB
        ax.imshow(rgb)

        # Plot boundary contours in red
        contours = measure.find_contours(mask.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 100:
                ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)

        # Plot reference edge in blue
        for idx, row in edge_gdf.iterrows():
            coords = geo_to_pixel(row.geometry, transform)
            if coords:
                xs, ys = zip(*coords)
                ax.plot(xs, ys, 'b-', linewidth=2)

        ax.set_title(f'{title}\nCoverage: {coverage:.1f}%', fontsize=11, fontweight='bold')
        ax.axis('off')

        print(f"  Otsu={otsu_thresh:.4f}, Adjusted={adj_thresh:.4f}, Coverage={coverage:.1f}%")

    # Legend
    legend_elements = [
        Line2D([0], [0], color='r', linewidth=2, label='Detected'),
        Line2D([0], [0], color='b', linewidth=2, label='Reference')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    plot_threshold_comparison()
