#!/usr/bin/env python3
"""
Landsat 8 SAVI Downloader

Downloads Landsat 8 imagery from Google Earth Engine and calculates SAVI
(Soil Adjusted Vegetation Index) for a specified study area.

Features:
- Query scenes by date range and cloud cover
- Interactive scene selection
- Download as GeoTIFF with all spectral bands + SAVI

Usage:
    python savi_downloader.py              # Interactive mode
    python savi_downloader.py --list-only  # Just list available scenes
    python savi_downloader.py --year 2023  # Filter to specific year
"""

import ee
import geemap
import argparse
import os

# Initialize Earth Engine with project (required for authentication)
GEE_PROJECT_ID = 'ee-faranido'

# SAVI parameters
SAVI_L_FACTOR = 0.5
CLOUD_THRESHOLD = 20
OUTPUT_DIR = './data'

# Study area polygon coordinates - multiple regions supported
STUDY_AREAS = {
    'beer_sheva': [
        [34.651, 31.83408468053888],
        [34.651, 31.10789572052074],
        [35.355, 31.10789572052074],
        [35.355, 31.83408468053888]
    ],
    'algeria': [
        # Shrunk to fit within single Landsat tile (path 196)
        # Original spanned paths 195-196, causing partial coverage
        [3.68, 36.17],
        [3.00, 36.17],
        [3.00, 35.50],
        [3.68, 35.50]
    ]
}

# Date ranges: Late September to early October for 2021, 2022, 2023
DATE_RANGES = [
    ('2021-09-20', '2021-10-15'),
    ('2022-09-20', '2022-10-15'),
    ('2023-09-20', '2023-10-15'),
]

# Landsat 8 band mapping
LANDSAT8_BANDS = {
    'SR_B1': 'Coastal',
    'SR_B2': 'Blue',
    'SR_B3': 'Green',
    'SR_B4': 'Red',
    'SR_B5': 'NIR',
    'SR_B6': 'SWIR1',
    'SR_B7': 'SWIR2',
}

# Cloud cover threshold for pixel-level filtering within polygon (percentage)
POLYGON_CLOUD_THRESHOLD = 15


def initialize_gee():
    """Initialize Earth Engine with project ID."""
    ee.Initialize(project=GEE_PROJECT_ID)


def get_study_area(region='beer_sheva'):
    """Create the study area polygon geometry.

    Args:
        region: Name of the region ('beer_sheva' or 'algeria')

    Returns:
        ee.Geometry.Polygon for the selected region
    """
    if region not in STUDY_AREAS:
        raise ValueError(f"Unknown region: {region}. Choose from: {list(STUDY_AREAS.keys())}")
    return ee.Geometry.Polygon([STUDY_AREAS[region]])


def calculate_cloud_percentage(image, geometry):
    """Calculate cloud cover percentage within the study area polygon.

    Uses QA_PIXEL band to identify cloud and cloud shadow pixels.
    Landsat Collection 2 QA_PIXEL bits:
    - Bit 3: Cloud (1 = cloud)
    - Bit 4: Cloud Shadow (1 = shadow)

    Returns:
        Cloud cover percentage (0-100) within the polygon
    """
    # Get QA_PIXEL band
    qa = image.select('QA_PIXEL')

    # Cloud bit is bit 3 (value 8), Cloud shadow is bit 4 (value 16)
    cloud_bit = 1 << 3  # 8
    shadow_bit = 1 << 4  # 16

    # Create cloud mask (1 = cloud or shadow, 0 = clear)
    cloud_mask = qa.bitwiseAnd(cloud_bit).neq(0).Or(
        qa.bitwiseAnd(shadow_bit).neq(0)
    )

    # Calculate percentage of cloudy pixels in the polygon
    stats = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    )

    # Mean of binary mask * 100 = percentage
    cloud_pct = ee.Number(stats.get('QA_PIXEL')).multiply(100)

    return cloud_pct


def query_landsat_scenes(geometry):
    """Query Landsat 8 Collection 2 Tier 1 Level 2 imagery.

    Filtering criteria:
    - Dataset: LANDSAT/LC08/C02/T1_L2
    - Date range: Sep 20 - Oct 15 for years 2021, 2022, 2023
    - Cloud cover: CLOUD_COVER_LAND < 20
    - Bounds: Study area polygon

    Returns:
        Filtered ee.ImageCollection
    """
    landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    # Build date filter for all three years
    date_filters = [ee.Filter.date(start, end) for start, end in DATE_RANGES]
    date_filter = ee.Filter.Or(*date_filters)

    collection = (landsat8
        .filterBounds(geometry)
        .filter(date_filter)
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', CLOUD_THRESHOLD)))

    return collection


def get_scene_metadata(collection, geometry):
    """Extract metadata from each scene for display.

    Returns list of dicts with:
    - scene_id: System index
    - date: Acquisition date
    - cloud_cover: CLOUD_COVER_LAND percentage (scene level)
    - polygon_cloud: Cloud percentage within study polygon (pixel level)
    - path: WRS path
    - row: WRS row
    """
    count = collection.size().getInfo()

    if count == 0:
        return []

    # Get list of images
    img_list = collection.toList(count)

    metadata = []
    for i in range(count):
        img = ee.Image(img_list.get(i))

        # Calculate pixel-level cloud cover within polygon
        polygon_cloud = calculate_cloud_percentage(img, geometry).getInfo()

        info = {
            'scene_id': img.get('system:index').getInfo(),
            'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo(),
            'cloud_cover': img.get('CLOUD_COVER_LAND').getInfo(),
            'polygon_cloud': polygon_cloud,
            'path': img.get('WRS_PATH').getInfo(),
            'row': img.get('WRS_ROW').getInfo(),
        }
        metadata.append(info)
        print(f"  Scene {i+1}/{count}: {info['date']} - Scene cloud: {info['cloud_cover']:.1f}%, Polygon cloud: {polygon_cloud:.1f}%")

    # Sort by date
    metadata.sort(key=lambda x: x['date'])

    return metadata


def group_scenes_by_date(metadata, require_full_coverage=True, max_polygon_cloud=None, main_tile=None):
    """Group scenes by acquisition date.

    Args:
        metadata: List of scene metadata
        require_full_coverage: If True, only return dates with main tile
        max_polygon_cloud: If set, filter out dates with polygon cloud cover above this threshold
        main_tile: Tuple of (path, row) for the main tile. If None, full coverage check is disabled.

    Returns list of dicts with:
    - date: Acquisition date
    - scenes: List of scene metadata for that date
    - avg_cloud: Average cloud cover across scenes (scene-level)
    - avg_polygon_cloud: Average cloud cover within polygon (pixel-level)
    - paths_rows: List of path/row combinations
    - has_full_coverage: Whether main tile is present
    """
    from collections import defaultdict

    if max_polygon_cloud is None:
        max_polygon_cloud = POLYGON_CLOUD_THRESHOLD

    date_groups = defaultdict(list)
    for scene in metadata:
        date_groups[scene['date']].append(scene)

    grouped = []
    for date, scenes in sorted(date_groups.items()):
        avg_cloud = sum(s['cloud_cover'] for s in scenes) / len(scenes)
        avg_polygon_cloud = sum(s['polygon_cloud'] for s in scenes) / len(scenes)
        paths_rows = [f"{s['path']}/{s['row']}" for s in scenes]

        # Check if main tile is present (if main_tile is specified)
        has_main_tile = False
        if main_tile is not None:
            has_main_tile = any(
                s['path'] == main_tile[0] and s['row'] == main_tile[1]
                for s in scenes
            )

        # Skip dates without full coverage if required and main_tile is specified
        if require_full_coverage and main_tile is not None and not has_main_tile:
            continue

        # Skip dates with too much cloud cover in the polygon
        if avg_polygon_cloud > max_polygon_cloud:
            print(f"  Skipping {date}: Polygon cloud {avg_polygon_cloud:.1f}% > {max_polygon_cloud}% threshold")
            continue

        grouped.append({
            'date': date,
            'scenes': scenes,
            'avg_cloud': avg_cloud,
            'avg_polygon_cloud': avg_polygon_cloud,
            'paths_rows': paths_rows,
            'scene_count': len(scenes),
            'has_full_coverage': has_main_tile
        })

    return grouped


def create_mosaic_for_date(collection, date_str, geometry):
    """Create a mosaic of all scenes for a given date.

    Combines multiple path/row scenes into one complete image.
    """
    # Filter collection to specific date
    date_start = ee.Date(date_str)
    date_end = date_start.advance(1, 'day')

    day_collection = collection.filterDate(date_start, date_end)

    # Create mosaic (combines all scenes, later pixels overwrite earlier)
    mosaic = day_collection.mosaic()

    return mosaic


def create_median_composite(collection, geometry, year=None):
    """Create a median composite from all scenes in collection.

    Uses median reducer to combine multiple dates, filling gaps
    where individual scenes have no data.

    Args:
        collection: ee.ImageCollection to composite
        geometry: Study area geometry
        year: Optional year to filter to (e.g., 2023)

    Returns:
        ee.Image median composite
    """
    if year:
        # Filter to specific year's date range
        start_date = f'{year}-09-01'
        end_date = f'{year}-10-31'
        collection = collection.filterDate(start_date, end_date)

    # Apply cloud masking before compositing
    def mask_clouds(image):
        qa = image.select('QA_PIXEL')
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(shadow_bit).eq(0)
        )
        return image.updateMask(mask)

    masked_collection = collection.map(mask_clouds)

    # Create median composite
    composite = masked_collection.median()

    return composite


def prepare_export_image(image, geometry):
    """Prepare image for export with scaled bands and SAVI.

    Apply scale factors and calculate SAVI.

    Returns image with:
    - All spectral bands (SR_B1-SR_B7) scaled to surface reflectance
    - SAVI band calculated from scaled values
    """
    # Select and scale optical bands
    optical_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    optical = image.select(optical_bands).multiply(0.0000275).add(-0.2)

    # Calculate SAVI from scaled bands
    # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    L = SAVI_L_FACTOR
    savi = optical.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
        {
            'NIR': optical.select('SR_B5'),
            'RED': optical.select('SR_B4'),
            'L': L
        }
    ).rename('SAVI')

    # Combine scaled bands with SAVI
    export_image = optical.addBands(savi)

    # Clip to study area
    export_image = export_image.clip(geometry)

    return export_image


def download_image(image, filename, region, scale=30):
    """Download image to local GeoTIFF using geemap.

    Args:
        image: ee.Image to download
        filename: Output filename
        region: ee.Geometry for the region of interest
        scale: Resolution in meters (default 30m for Landsat)

    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    try:
        geemap.download_ee_image(
            image,
            filename=filename,
            region=region,
            scale=scale,
            crs='EPSG:4326'
        )
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def write_summary_file(output_dir, downloaded_files, study_area_coords):
    """Write a summary file with metadata for all downloaded files.

    Args:
        output_dir: Directory where files are saved
        downloaded_files: List of dicts with file info (filename, date, cloud_cover, paths_rows)
        study_area_coords: The polygon coordinates used
    """
    import csv
    from datetime import datetime

    summary_path = os.path.join(output_dir, 'download_summary.txt')
    csv_path = os.path.join(output_dir, 'download_summary.csv')

    # Write text summary
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Landsat 8 SAVI Download Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: LANDSAT/LC08/C02/T1_L2 (Landsat 8 Collection 2 Tier 1 Level 2)\n")
        f.write(f"Scene Cloud Filter: < {CLOUD_THRESHOLD}%\n")
        f.write(f"Polygon Cloud Filter: < {POLYGON_CLOUD_THRESHOLD}% (pixel-level within study area)\n\n")

        f.write("Study Area Polygon:\n")
        for i, coord in enumerate(study_area_coords):
            f.write(f"  Point {i+1}: [{coord[0]}, {coord[1]}]\n")
        f.write("\n")

        f.write("SAVI Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)\n")
        f.write(f"L Factor: {SAVI_L_FACTOR}\n")
        f.write("Bands: NIR = SR_B5, Red = SR_B4\n\n")

        f.write("Output Bands:\n")
        f.write("  1. SR_B1 (Coastal) - scaled\n")
        f.write("  2. SR_B2 (Blue) - scaled\n")
        f.write("  3. SR_B3 (Green) - scaled\n")
        f.write("  4. SR_B4 (Red) - scaled\n")
        f.write("  5. SR_B5 (NIR) - scaled\n")
        f.write("  6. SR_B6 (SWIR1) - scaled\n")
        f.write("  7. SR_B7 (SWIR2) - scaled\n")
        f.write("  8. SAVI - calculated\n\n")

        f.write("-" * 100 + "\n")
        f.write("Downloaded Files:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Filename':<30} | {'Date':<12} | {'Scene Cloud':>11} | {'Polygon Cloud':>13} | {'Size (MB)':>10} | {'Tiles':<15}\n")
        f.write("-" * 100 + "\n")

        for file_info in downloaded_files:
            polygon_cloud = file_info.get('polygon_cloud', file_info['cloud_cover'])
            f.write(f"{file_info['filename']:<30} | {file_info['date']:<12} | "
                    f"{file_info['cloud_cover']:>10.1f}% | {polygon_cloud:>12.1f}% | "
                    f"{file_info['size_mb']:>10.1f} | {file_info['paths_rows']:<15}\n")

        f.write("-" * 100 + "\n")
        f.write(f"Total: {len(downloaded_files)} files\n")

    # Write CSV for easy import
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'date', 'scene_cloud_percent', 'polygon_cloud_percent', 'size_mb', 'tiles'])
        for file_info in downloaded_files:
            polygon_cloud = file_info.get('polygon_cloud', file_info['cloud_cover'])
            writer.writerow([
                file_info['filename'],
                file_info['date'],
                round(file_info['cloud_cover'], 1),
                round(polygon_cloud, 1),
                round(file_info['size_mb'], 1),
                file_info['paths_rows']
            ])

    return summary_path, csv_path


def display_available_scenes(metadata_list):
    """Display formatted table of available scenes."""
    print("\n" + "=" * 75)
    print("Available Landsat 8 Scenes (Sep 20 - Oct 15, 2021-2023, Cloud < 20%)")
    print("=" * 75)
    print(f"{'Idx':>4} | {'Scene ID':<28} | {'Date':<10} | {'Cloud %':>8} | {'Path':>4} | {'Row':>4}")
    print("-" * 75)

    for idx, scene in enumerate(metadata_list, 1):
        print(f"{idx:>4} | {scene['scene_id']:<28} | {scene['date']:<10} | "
              f"{scene['cloud_cover']:>8.1f} | {scene['path']:>4} | {scene['row']:>4}")

    print("-" * 75)
    print(f"Total: {len(metadata_list)} scenes found\n")


def display_available_dates(date_groups):
    """Display formatted table of available dates (mosaicked)."""
    print("\n" + "=" * 95)
    print("Available Dates (scenes will be mosaicked to cover full study area)")
    print("=" * 95)
    print(f"{'Idx':>4} | {'Date':<12} | {'Scene Cloud':>11} | {'Polygon Cloud':>13} | {'Scenes':>6} | {'Path/Row':<20}")
    print("-" * 95)

    for idx, group in enumerate(date_groups, 1):
        paths_str = ', '.join(group['paths_rows'])
        polygon_cloud = group.get('avg_polygon_cloud', group['avg_cloud'])
        print(f"{idx:>4} | {group['date']:<12} | {group['avg_cloud']:>10.1f}% | {polygon_cloud:>12.1f}% | {group['scene_count']:>6} | {paths_str:<20}")

    print("-" * 95)
    print(f"Total: {len(date_groups)} unique dates")
    print(f"Note: 'Polygon Cloud' is pixel-level cloud cover within study area (more accurate)\n")


def get_user_selection(num_scenes):
    """Get user selection of scenes to download.

    Supports:
    - Single index: "1"
    - Multiple indices: "1,3,5"
    - Range: "1-5"
    - All: "all" or "*"
    - Quit: "q"

    Returns list of 0-based indices.
    """
    while True:
        print("Select scenes to download:")
        print("  - Enter numbers separated by commas: 1,3,5")
        print("  - Enter range: 1-5")
        print("  - Enter 'all' for all scenes")
        print("  - Enter 'q' to quit")

        choice = input("\nYour selection: ").strip().lower()

        if choice == 'q':
            return []

        if choice in ('all', '*'):
            return list(range(num_scenes))

        try:
            indices = []
            for part in choice.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start - 1, end))  # Convert to 0-based
                else:
                    indices.append(int(part) - 1)  # Convert to 0-based

            # Validate indices
            if all(0 <= idx < num_scenes for idx in indices):
                return sorted(set(indices))
            else:
                print(f"Error: Index out of range (1-{num_scenes})")
        except ValueError:
            print("Error: Invalid input format")


def main():
    """Main execution workflow."""
    parser = argparse.ArgumentParser(
        description='Download Landsat 8 SAVI imagery from Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python savi_downloader.py                    # Interactive mode
  python savi_downloader.py --list-only        # Just list available scenes
  python savi_downloader.py --output ./output  # Custom output directory
  python savi_downloader.py --year 2023        # Filter to specific year
        '''
    )
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--list-only', '-l', action='store_true',
                        help='Only list available scenes, do not download')
    parser.add_argument('--year', '-y', type=int, choices=[2021, 2022, 2023],
                        help='Filter to specific year')
    parser.add_argument('--scale', '-s', type=int, default=30,
                        help='Resolution in meters (default: 30)')
    parser.add_argument('--region', '-r', type=str, default='beer_sheva',
                        choices=['beer_sheva', 'algeria'],
                        help='Study area region (default: beer_sheva)')
    parser.add_argument('--composite', '-c', action='store_true',
                        help='Create median composite from all scenes (fills gaps)')
    parser.add_argument('--composite-year', type=int, choices=[2021, 2022, 2023],
                        help='Year for composite (required with --composite)')

    args = parser.parse_args()

    print("=" * 50)
    print("Landsat 8 SAVI Downloader")
    print("=" * 50)

    # Initialize
    print("\nInitializing Google Earth Engine...")
    initialize_gee()

    # Get study area
    geometry = get_study_area(args.region)
    study_area_coords = STUDY_AREAS[args.region]
    print(f"Region: {args.region}")
    print(f"Study area: {study_area_coords[0]} to {study_area_coords[2]}")

    # Query scenes
    print("\nQuerying available scenes...")
    collection = query_landsat_scenes(geometry)

    # Get metadata
    print("Fetching scene metadata and calculating polygon cloud cover...")
    print("(This calculates actual cloud % within your study area)")
    metadata = get_scene_metadata(collection, geometry)

    if not metadata:
        print("No scenes found matching criteria.")
        return

    # Filter by year if specified
    if args.year:
        metadata = [s for s in metadata if s['date'].startswith(str(args.year))]
        if not metadata:
            print(f"No scenes found for year {args.year}.")
            return

    # Group scenes by date for mosaicking
    # Define main tile for each region (path/row that should cover most of the area)
    main_tiles = {
        'beer_sheva': (174, 38),
        'algeria': None  # No specific main tile required for Algeria
    }
    main_tile = main_tiles.get(args.region)
    date_groups = group_scenes_by_date(metadata, main_tile=main_tile)

    # Display available dates
    display_available_dates(date_groups)

    if args.list_only:
        return

    # Composite mode - create median composite from all scenes
    if args.composite:
        year = args.composite_year
        if not year:
            print("Error: --composite-year required with --composite")
            return

        print(f"\nCreating median composite for {year}...")
        print("  This combines all scenes to fill gaps from single-scene coverage")

        # Create composite
        composite = create_median_composite(collection, geometry, year=year)

        # Prepare for export
        print("  Preparing image (scaling bands, calculating SAVI)...")
        export_image = prepare_export_image(composite, geometry)

        # Download
        os.makedirs(args.output, exist_ok=True)
        filename = os.path.join(args.output, f"LC08_{args.region}_{year}_composite_SAVI.tif")
        print(f"  Downloading to: {filename}")
        print(f"  (This may take several minutes...)")

        success = download_image(export_image, filename, geometry, scale=args.scale)

        if success:
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"\n  Done! Size: {file_size_mb:.1f} MB")
        else:
            print(f"  Failed to download.")

        print("\n" + "=" * 50)
        print("Composite download complete!")
        print("=" * 50)
        return

    # Get user selection
    selected_indices = get_user_selection(len(date_groups))

    if not selected_indices:
        print("No dates selected. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Download selected dates (mosaicked)
    print(f"\nDownloading {len(selected_indices)} date(s) to {args.output}/...")

    downloaded_files = []

    for i, idx in enumerate(selected_indices, 1):
        date_info = date_groups[idx]
        date_str = date_info['date']

        print(f"\n[{i}/{len(selected_indices)}] Processing date: {date_str}")
        print(f"  Mosaicking {date_info['scene_count']} scene(s): {', '.join(date_info['paths_rows'])}")

        # Create mosaic for this date
        mosaic = create_mosaic_for_date(collection, date_str, geometry)

        # Prepare for export (scale bands + calculate SAVI)
        print("  Preparing image (scaling bands, calculating SAVI)...")
        export_image = prepare_export_image(mosaic, geometry)

        # Download locally
        filename = os.path.join(args.output, f"LC08_{args.region}_{date_str.replace('-', '')}_SAVI.tif")
        print(f"  Downloading to: {filename}")
        print(f"  (This may take several minutes for large regions...)")

        success = download_image(export_image, filename, geometry, scale=args.scale)

        if success:
            print(f"  Done!")
            # Get file size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            downloaded_files.append({
                'filename': os.path.basename(filename),
                'date': date_str,
                'cloud_cover': date_info['avg_cloud'],
                'polygon_cloud': date_info.get('avg_polygon_cloud', date_info['avg_cloud']),
                'size_mb': file_size_mb,
                'paths_rows': ', '.join(date_info['paths_rows'])
            })
        else:
            print(f"  Failed to download. Try with --scale 60 or larger region may need tiling.")

    # Write summary files
    if downloaded_files:
        print("\nWriting summary files...")
        summary_path, csv_path = write_summary_file(args.output, downloaded_files, study_area_coords)
        print(f"  Summary: {summary_path}")
        print(f"  CSV: {csv_path}")

    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
