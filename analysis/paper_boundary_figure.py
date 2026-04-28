"""Generate paper figure comparing boundary detection methods."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from scipy.ndimage import binary_closing, binary_opening
import geopandas as gpd


def load_rgb_image(tif_path, bands=[4, 3, 2], percentile=(2, 98), target_shape=None):
    """Load and normalize RGB from GeoTIFF."""
    with rasterio.open(tif_path) as src:
        rgb = np.stack([src.read(b) for b in bands], axis=-1).astype(float)
    for i in range(3):
        valid = rgb[:, :, i][rgb[:, :, i] > 0]
        if len(valid) > 0:
            p_low, p_high = np.percentile(valid, percentile)
            rgb[:, :, i] = np.clip((rgb[:, :, i] - p_low) / (p_high - p_low), 0, 1)
    if target_shape is not None:
        rgb = resize(rgb, (target_shape[0], target_shape[1], 3), anti_aliasing=True)
    return rgb


def compute_kmeans_on_ndvi(ndvi_path, k=2):
    """Compute K-means clustering on NDVI values."""
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)

    h, w = ndvi.shape
    flat_ndvi = ndvi.reshape(-1, 1)
    valid = np.isfinite(flat_ndvi).flatten() & (flat_ndvi.flatten() != 0)

    # Fit K-means on valid pixels
    valid_idx = np.where(valid)[0]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(flat_ndvi[valid])

    # Predict all pixels
    labels = np.full(h * w, -1)
    labels[valid] = kmeans.predict(flat_ndvi[valid])
    labels = labels.reshape(h, w)

    # Identify vegetation cluster (highest centroid = highest NDVI)
    veg_cluster = np.argmax(kmeans.cluster_centers_.flatten())

    veg_mask = labels == veg_cluster
    print(f"K-means k={k} on NDVI: vegetation cluster={veg_cluster}, coverage={veg_mask.mean()*100:.1f}%")

    return veg_mask


def compute_gmm_on_ndvi(ndvi_path, k=2):
    """Compute GMM clustering on NDVI values."""
    from sklearn.mixture import GaussianMixture

    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)

    h, w = ndvi.shape
    flat_ndvi = ndvi.reshape(-1, 1)
    valid = np.isfinite(flat_ndvi).flatten() & (flat_ndvi.flatten() != 0)

    # Fit GMM on valid pixels
    valid_idx = np.where(valid)[0]
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(flat_ndvi[valid])

    # Predict probabilities for all pixels
    probs = np.zeros((h * w, k))
    probs[valid] = gmm.predict_proba(flat_ndvi[valid])

    # Identify vegetation component (highest mean)
    veg_component = np.argmax(gmm.means_.flatten())

    veg_prob = probs[:, veg_component].reshape(h, w)
    veg_mask = veg_prob > 0.5

    print(f"GMM k={k} on NDVI: vegetation component={veg_component}, coverage={veg_mask.mean()*100:.1f}%")

    return veg_mask


def plot_comparison_figure(region='beer_sheva', output_path=None, show_reference=True):
    """Create 4-panel boundary comparison: RGB + K-means + GMM + Ours.

    Args:
        region: Region name ('beer_sheva' or 'algeria')
        output_path: Optional output path for the figure
        show_reference: Whether to show reference boundary lines (default True)
    """

    if region == 'beer_sheva':
        ndvi_path = 'data/beer_sheva/LC08_20221001_SAVI.tif'
        embed_path = 'data/beer_sheva/boundary/boundary_embedding.npz'
        reference_shps = [
            ('data/beer_sheva/edge/phyto-line.shp', 'Phytogeographic', 'blue'),
            ('data/beer_sheva/edge250/250mm.shp', '250mm Isohyet', 'cyan'),
        ]
        title_prefix = 'Beer Sheva'
        if output_path is None:
            output_path = 'outputs/paper_boundary_beer_sheva.png'
    elif region == 'algeria':
        ndvi_path = 'data/algeria/LC08_algeria_2023_composite_SAVI.tif'
        embed_path = 'data/algeria/boundary/boundary_embedding.npz'
        reference_shps = []  # No reference for Algeria
        title_prefix = 'Algeria'
        if output_path is None:
            output_path = 'outputs/paper_boundary_algeria.png'
    else:
        raise ValueError(f"Unknown region: {region}")

    # Load our method boundary
    embed_data = np.load(embed_path)
    embed_mask = embed_data['smoothed_mask'].astype(bool)

    # Compute K-means k=2 on NDVI
    kmeans_mask = compute_kmeans_on_ndvi(ndvi_path, k=2)

    # Compute GMM k=2 on NDVI
    gmm_mask = compute_gmm_on_ndvi(ndvi_path, k=2)

    # Use kmeans/gmm shape as target (they come from NDVI which is larger)
    target_shape = kmeans_mask.shape

    # Resize embed_mask to match
    if embed_mask.shape != target_shape:
        embed_mask = resize(embed_mask.astype(float), target_shape, order=0) > 0.5

    # Resize gmm_mask if needed
    if gmm_mask.shape != target_shape:
        gmm_mask = resize(gmm_mask.astype(float), target_shape, order=0) > 0.5

    # Load satellite RGB
    rgb = load_rgb_image(ndvi_path, target_shape=target_shape)

    # Load reference shapefiles and convert to pixel coordinates
    reference_data = []  # List of (pixels, name, color)
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        orig_shape = (src.height, src.width)

    if not show_reference:
        reference_shps = []  # Skip loading references

    for shp_path, name, color in reference_shps:
        try:
            gdf = gpd.read_file(shp_path)

            # Convert geo coords to pixel coords
            pixel_lines = []
            for geom in gdf.geometry:
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                elif geom.geom_type == 'MultiLineString':
                    coords = []
                    for line in geom.geoms:
                        coords.extend(list(line.coords))
                else:
                    continue

                pixel_coords = []
                for x, y in coords:
                    col, row = ~transform * (x, y)
                    # Scale to target shape
                    row_scaled = row * target_shape[0] / orig_shape[0]
                    col_scaled = col * target_shape[1] / orig_shape[1]
                    pixel_coords.append((row_scaled, col_scaled))
                pixel_lines.append(pixel_coords)

            reference_data.append((pixel_lines, name, color))
            print(f"Loaded {name}: {len(pixel_lines)} lines")
        except Exception as e:
            print(f"Warning: Could not load {shp_path}: {e}")

    # Create figure - 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # Colors: green for vegetation, brown for desert
    green = np.array([34, 139, 34, 150]) / 255.0  # Forest green with alpha
    brown = np.array([210, 180, 140, 150]) / 255.0  # Tan/desert brown with alpha

    def plot_colored_mask(ax, rgb_img, mask, title, ref_data=None):
        # Show satellite background
        ax.imshow(rgb_img)

        # Create RGBA overlay: green for vegetation, brown for desert
        h, w = mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[mask] = green  # vegetation = green
        overlay[~mask] = brown  # desert = brown

        ax.imshow(overlay)

        # Smooth the mask for cleaner boundary
        from scipy.ndimage import binary_closing, binary_opening
        smooth_mask = binary_closing(mask, iterations=2)
        smooth_mask = binary_opening(smooth_mask, iterations=2)

        # Red boundary line (detected)
        contours = measure.find_contours(smooth_mask.astype(float), 0.5)
        for contour in contours:
            if len(contour) > 100:
                ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)

        # Reference lines
        if ref_data:
            for pixel_lines, name, color in ref_data:
                for line in pixel_lines:
                    rows = [p[0] for p in line]
                    cols = [p[1] for p in line]
                    ax.plot(cols, rows, '-', color=color, linewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Panel A: Satellite image (with reference lines if enabled)
    axes[0].imshow(rgb)
    if reference_data:
        for pixel_lines, name, color in reference_data:
            for line in pixel_lines:
                rows = [p[0] for p in line]
                cols = [p[1] for p in line]
                axes[0].plot(cols, rows, '-', color=color, linewidth=2)
    axes[0].set_title('(a) Landsat True Color', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel B: K-means (invert mask)
    plot_colored_mask(axes[1], rgb, ~kmeans_mask, '(b) K-means', reference_data)

    # Panel C: GMM (invert mask)
    plot_colored_mask(axes[2], rgb, ~gmm_mask, '(c) GMM', reference_data)

    # Panel D: Ours
    plot_colored_mask(axes[3], rgb, embed_mask, '(d) Ours', reference_data)

    # Add legend only if reference lines are shown
    if reference_data:
        legend_elements = [
            Line2D([0], [0], color='r', linewidth=2, label='Detected'),
        ]
        for _, name, color in reference_data:
            legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=name))
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), fontsize=11, frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for legend
    else:
        plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper figure comparing boundary detection methods')
    parser.add_argument('region', nargs='?', default=None, help='Region name (beer_sheva or algeria)')
    parser.add_argument('--no-reference', action='store_true', help='Hide reference boundary lines')
    parser.add_argument('-o', '--output', default=None, help='Output path for the figure')

    args = parser.parse_args()

    show_ref = not args.no_reference

    if args.region:
        plot_comparison_figure(region=args.region, output_path=args.output, show_reference=show_ref)
    else:
        # Generate both figures
        print("=== Beer Sheva ===")
        plot_comparison_figure(region='beer_sheva', show_reference=show_ref)
        print("\n=== Algeria ===")
        plot_comparison_figure(region='algeria', show_reference=show_ref)
