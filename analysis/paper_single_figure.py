"""Generate single paper figure with low-opacity overlays and boundary lines."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
from skimage import measure
from skimage.transform import resize
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


def generate_paper_figure(region='beer_sheva', output_path=None):
    """Generate single figure with low-opacity green/brown overlays and boundary lines."""

    if region == 'beer_sheva':
        ndvi_path = 'data/beer_sheva/LC08_20221001_SAVI.tif'
        embed_path = 'data/beer_sheva/boundary/boundary_embedding.npz'
        reference_shps = [
            ('data/beer_sheva/edge/phyto-line.shp', 'Phytogeographic', 'blue'),
        ]
        if output_path is None:
            output_path = 'outputs/paper_figure_beer_sheva.png'
    elif region == 'algeria':
        ndvi_path = 'data/algeria/LC08_algeria_2023_composite_SAVI.tif'
        embed_path = 'data/algeria/boundary/boundary_embedding.npz'
        reference_shps = []
        if output_path is None:
            output_path = 'outputs/paper_figure_algeria.png'
    else:
        raise ValueError(f"Unknown region: {region}")

    # Load our method boundary
    embed_data = np.load(embed_path)
    embed_mask = embed_data['smoothed_mask'].astype(bool)

    # Get target shape from NDVI
    with rasterio.open(ndvi_path) as src:
        target_shape = (src.height, src.width)
        transform = src.transform

    # Resize embed_mask to match
    if embed_mask.shape != target_shape:
        embed_mask = resize(embed_mask.astype(float), target_shape, order=0) > 0.5

    # Load satellite RGB
    rgb = load_rgb_image(ndvi_path, target_shape=target_shape)

    # Load reference shapefiles
    reference_data = []
    for shp_path, name, color in reference_shps:
        try:
            gdf = gpd.read_file(shp_path)
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
                    pixel_coords.append((row, col))
                pixel_lines.append(pixel_coords)

            reference_data.append((pixel_lines, name, color))
            print(f"Loaded {name}: {len(pixel_lines)} lines")
        except Exception as e:
            print(f"Warning: Could not load {shp_path}: {e}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

    # Show satellite background
    ax.imshow(rgb)

    # Create low-opacity overlay: green for vegetation, brown for desert
    h, w = embed_mask.shape
    overlay = np.zeros((h, w, 4))

    # Green for vegetation, brown for desert (matching reference image style)
    green = np.array([34, 139, 34, 50]) / 255.0  # Alpha ≈ 0.20 (less green)
    brown = np.array([190, 120, 80, 120]) / 255.0  # More brown, alpha ≈ 0.47

    overlay[embed_mask] = green
    overlay[~embed_mask] = brown

    ax.imshow(overlay)

    # Smooth the mask for cleaner boundary contour
    smooth_mask = binary_closing(embed_mask, iterations=3)
    smooth_mask = binary_opening(smooth_mask, iterations=3)

    # Red boundary line (our detected boundary)
    contours = measure.find_contours(smooth_mask.astype(float), 0.5)
    for contour in contours:
        if len(contour) > 100:
            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2.5, label='Ours')

    # Blue reference lines (clipped to ROI with small margin)
    margin = 5  # Allow small margin outside bounds
    for pixel_lines, name, color in reference_data:
        for i, line in enumerate(pixel_lines):
            rows = [p[0] for p in line]
            cols = [p[1] for p in line]
            # Clip to image bounds with margin
            clipped_rows = []
            clipped_cols = []
            for r, c in zip(rows, cols):
                if -margin <= r < h + margin and -margin <= c < w + margin:
                    # Clamp to actual bounds for plotting
                    clipped_rows.append(max(0, min(h-1, r)))
                    clipped_cols.append(max(0, min(w-1, c)))
                else:
                    # Draw segment if we have points, then reset
                    if len(clipped_rows) > 1:
                        ax.plot(clipped_cols, clipped_rows, '-', color=color, linewidth=2.5)
                    clipped_rows = []
                    clipped_cols = []
            # Draw remaining segment
            if len(clipped_rows) > 1:
                ax.plot(clipped_cols, clipped_rows, '-', color=color, linewidth=2.5)

    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Inverted for image coordinates

    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper figure with boundary overlay')
    parser.add_argument('region', nargs='?', default='beer_sheva', help='Region (beer_sheva or algeria)')
    parser.add_argument('-o', '--output', default=None, help='Output path')

    args = parser.parse_args()
    generate_paper_figure(region=args.region, output_path=args.output)
