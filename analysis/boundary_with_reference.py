"""Generate boundary comparison with reference edge overlay."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
from rasterio.plot import show
import geopandas as gpd
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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


def compute_kmeans_on_ndvi(ndvi_path, k=2):
    """Compute K-means clustering on NDVI values."""
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
        transform = src.transform

    h, w = ndvi.shape
    flat_ndvi = ndvi.reshape(-1, 1)
    valid = np.isfinite(flat_ndvi).flatten() & (flat_ndvi.flatten() != 0)

    valid_idx = np.where(valid)[0]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(flat_ndvi[valid])

    labels = np.full(h * w, -1)
    labels[valid] = kmeans.predict(flat_ndvi[valid])
    labels = labels.reshape(h, w)

    veg_cluster = np.argmax(kmeans.cluster_centers_.flatten())
    veg_mask = labels == veg_cluster

    return veg_mask, transform


def compute_gmm_on_ndvi(ndvi_path, k=2):
    """Compute GMM clustering on NDVI values."""
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)

    h, w = ndvi.shape
    flat_ndvi = ndvi.reshape(-1, 1)
    valid = np.isfinite(flat_ndvi).flatten() & (flat_ndvi.flatten() != 0)

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(flat_ndvi[valid])

    probs = np.zeros((h * w, k))
    probs[valid] = gmm.predict_proba(flat_ndvi[valid])

    veg_component = np.argmax(gmm.means_.flatten())
    veg_prob = probs[:, veg_component].reshape(h, w)
    veg_mask = veg_prob > 0.5

    return veg_mask


def plot_with_reference(output_path='outputs/boundary_with_reference.png'):
    """Create visualization with K-means, GMM, Ours, and reference edge."""

    ndvi_path = 'data/beer_sheva/LC08_20221001_SAVI.tif'
    edge_path = 'data/beer_sheva/edge/phyto-line.shp'
    embed_path = 'data/beer_sheva/boundary/boundary_embedding.npz'

    # Load reference edge
    edge_gdf = gpd.read_file(edge_path)

    # Load RGB and get transform
    rgb, transform, crs = load_rgb_image(ndvi_path)

    # Reproject edge to match raster CRS if needed
    if edge_gdf.crs != crs:
        edge_gdf = edge_gdf.to_crs(crs)

    # Load our method boundary
    embed_data = np.load(embed_path)
    embed_mask = embed_data['smoothed_mask'].astype(bool)

    # Compute K-means and GMM
    kmeans_mask, _ = compute_kmeans_on_ndvi(ndvi_path, k=2)
    gmm_mask = compute_gmm_on_ndvi(ndvi_path, k=2)

    # Resize embed_mask to match
    target_shape = kmeans_mask.shape
    if embed_mask.shape != target_shape:
        embed_mask = resize(embed_mask.astype(float), target_shape, order=0) > 0.5

    # Create figure - 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # Helper to convert geo coordinates to pixel coordinates
    def geo_to_pixel(geom, transform):
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

    def plot_panel(ax, rgb_img, mask, title, show_reference=True):
        ax.imshow(rgb_img)

        # Boundary contours
        contours = measure.find_contours(mask.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 100:
                ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, label='Detected')

        # Reference edge in blue
        if show_reference:
            for idx, row in edge_gdf.iterrows():
                coords = geo_to_pixel(row.geometry, transform)
                if coords:
                    xs, ys = zip(*coords)
                    ax.plot(xs, ys, 'b-', linewidth=2, label='Reference')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(0, rgb_img.shape[1])
        ax.set_ylim(rgb_img.shape[0], 0)

    # Panel A: RGB with reference only
    axes[0].imshow(rgb)
    for idx, row in edge_gdf.iterrows():
        coords = geo_to_pixel(row.geometry, transform)
        if coords:
            xs, ys = zip(*coords)
            axes[0].plot(xs, ys, 'b-', linewidth=2)
    axes[0].set_title('(a) Reference Edge', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel B: K-means (invert)
    plot_panel(axes[1], rgb, ~kmeans_mask, '(b) K-means')

    # Panel C: GMM (invert)
    plot_panel(axes[2], rgb, ~gmm_mask, '(c) GMM')

    # Panel D: Ours
    plot_panel(axes[3], rgb, embed_mask, '(d) Ours')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='r', linewidth=2, label='Detected'),
        Line2D([0], [0], color='b', linewidth=2, label='Reference')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    plot_with_reference()
