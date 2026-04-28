#!/usr/bin/env python3
"""
KML Region Downloader for Google Earth Engine Data

Downloads satellite data (SAVI and Google embeddings) for polygon regions
defined in KML files. Each polygon in the KML becomes a separate download.

Features:
- Read polygon geometries from KML files
- Download Landsat 8/9 SAVI data per region
- Download Google satellite embeddings per region
- Region-specific filenames based on KML names
- Support for both data types in single run

Usage:
    # Download both SAVI and embeddings for all regions
    python kml_region_downloader.py \\
        --kml data/beer_sheva/sub_regions.kml \\
        --savi --year 2022 \\
        --embeddings --embedding-year 2022 \\
        --output data/beer_sheva/

    # Download only SAVI
    python kml_region_downloader.py \\
        --kml data/beer_sheva/sub_regions.kml \\
        --savi --year 2022 \\
        --output data/beer_sheva/

    # Download only embeddings
    python kml_region_downloader.py \\
        --kml data/beer_sheva/sub_regions.kml \\
        --embeddings --embedding-year 2022 \\
        --output data/beer_sheva/
"""

import ee
import geemap
import geopandas as gpd
import argparse
import os
import re
from datetime import datetime
from pathlib import Path

# Initialize Earth Engine with project
GEE_PROJECT_ID = 'ee-faranido'

# SAVI parameters (from landsat_savi.py)
SAVI_L_FACTOR = 0.5
CLOUD_THRESHOLD = 20
POLYGON_CLOUD_THRESHOLD = 15

# Default output directory
OUTPUT_DIR = './data'


def initialize_gee():
    """Initialize Earth Engine with project ID."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Please run: earthengine authenticate")
        return False


def sanitize_region_name(name):
    """Convert region name to filesystem-safe string.

    Args:
        name: Original region name from KML

    Returns:
        Sanitized name suitable for filenames
    """
    # Replace spaces with underscores, remove special characters
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def read_kml_regions(kml_path):
    """Read polygon regions from KML file.

    Args:
        kml_path: Path to KML file

    Returns:
        List of tuples (region_name, geometry) where geometry is shapely Polygon
        Returns None if file cannot be read
    """
    try:
        # Read KML using geopandas
        gdf = gpd.read_file(kml_path)

        if len(gdf) == 0:
            print(f"Error: No features found in KML file: {kml_path}")
            return None

        regions = []
        for idx, row in gdf.iterrows():
            name = row.get('Name') or row.get('name') or f"region_{idx}"
            geometry = row.geometry

            # Only process polygon geometries
            if geometry.geom_type not in ['Polygon', 'MultiPolygon']:
                print(f"  Skipping non-polygon feature: {name} ({geometry.geom_type})")
                continue

            regions.append((name, geometry))

        return regions

    except Exception as e:
        print(f"Error reading KML file: {e}")
        return None


def shapely_to_ee_geometry(shapely_polygon):
    """Convert shapely Polygon to Earth Engine Geometry.

    Args:
        shapely_polygon: Shapely Polygon or MultiPolygon

    Returns:
        ee.Geometry.Polygon
    """
    # Handle MultiPolygon by taking the first polygon
    if shapely_polygon.geom_type == 'MultiPolygon':
        shapely_polygon = list(shapely_polygon.geoms)[0]

    # Extract coordinates - shapely uses (lon, lat) which is what GEE expects
    coords = list(shapely_polygon.exterior.coords)
    coords_list = [[coord[0], coord[1]] for coord in coords]

    return ee.Geometry.Polygon([coords_list])


def calculate_cloud_percentage(image, geometry):
    """Calculate cloud cover percentage within the study area polygon.

    Uses QA_PIXEL band to identify cloud and cloud shadow pixels.
    (Adapted from landsat_savi.py)

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


def query_landsat_for_region(geometry, year):
    """Query Landsat 8 imagery for a specific region and year.

    Args:
        geometry: ee.Geometry for the region
        year: Year to query (2021-2024)

    Returns:
        ee.ImageCollection filtered to region and date range
    """
    landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    # Query for late September to early October (good for vegetation)
    date_start = f'{year}-09-20'
    date_end = f'{year}-10-15'

    collection = (landsat8
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', CLOUD_THRESHOLD)))

    return collection


def select_best_scene_for_region(collection, geometry):
    """Select the best scene for a region based on cloud cover.

    Args:
        collection: ee.ImageCollection of Landsat scenes
        geometry: ee.Geometry for the region

    Returns:
        Tuple of (ee.Image, date_str, cloud_pct) or (None, None, None) if no scenes
    """
    count = collection.size().getInfo()

    if count == 0:
        return None, None, None

    # Get list of images and calculate polygon cloud cover for each
    img_list = collection.toList(count)

    best_image = None
    best_date = None
    best_cloud = 100.0

    print(f"    Evaluating {count} scene(s)...")

    for i in range(count):
        img = ee.Image(img_list.get(i))

        # Calculate pixel-level cloud cover within polygon
        try:
            polygon_cloud = calculate_cloud_percentage(img, geometry).getInfo()
            date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            print(f"      {date_str}: {polygon_cloud:.1f}% cloud in region")

            if polygon_cloud < best_cloud:
                best_cloud = polygon_cloud
                best_date = date_str
                best_image = img
        except Exception as e:
            print(f"      Error evaluating scene: {e}")
            continue

    return best_image, best_date, best_cloud


def prepare_savi_export_image(image, geometry):
    """Prepare Landsat image for SAVI export.

    Apply scale factors and calculate SAVI.
    (Adapted from landsat_savi.py)

    Returns:
        ee.Image with scaled bands and SAVI
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


def download_savi_for_region(region_name, geometry, year, output_dir, scale=30):
    """Download SAVI data for a specific region.

    Args:
        region_name: Name of the region (for filename)
        geometry: ee.Geometry for the region
        year: Year to download
        output_dir: Output directory path
        scale: Resolution in meters

    Returns:
        Dict with download info or None if failed
    """
    print(f"  Querying Landsat 8 for {region_name}...")

    # Query Landsat collection
    collection = query_landsat_for_region(geometry, year)

    # Select best scene
    image, date_str, cloud_pct = select_best_scene_for_region(collection, geometry)

    if image is None:
        print(f"    No suitable scenes found for {region_name}")
        return None

    print(f"    Selected: {date_str} ({cloud_pct:.1f}% cloud)")

    # Prepare for export
    print(f"    Preparing image (scaling bands, calculating SAVI)...")
    export_image = prepare_savi_export_image(image, geometry)

    # Create filename
    sanitized_name = sanitize_region_name(region_name)
    date_compact = date_str.replace('-', '')
    filename = os.path.join(output_dir, f"LC08_{sanitized_name}_{date_compact}_SAVI.tif")

    print(f"    Downloading to: {filename}")
    print(f"    (This may take several minutes...)")

    # Download
    try:
        geemap.download_ee_image(
            export_image,
            filename=filename,
            region=geometry,
            scale=scale,
            crs='EPSG:4326'
        )

        # Get file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"    Done! Size: {file_size_mb:.1f} MB")

        return {
            'type': 'savi',
            'region': region_name,
            'filename': os.path.basename(filename),
            'date': date_str,
            'cloud_pct': cloud_pct,
            'size_mb': file_size_mb
        }

    except Exception as e:
        print(f"    Download error: {e}")
        return None


def query_embedding_for_region(geometry, year):
    """Query Google Satellite Embedding for a specific region and year.

    Args:
        geometry: ee.Geometry for the region
        year: Year to query (2017-2024)

    Returns:
        ee.Image with 64 embedding bands
    """
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Filter to specific year
    date_start = f'{year}-01-01'
    date_end = f'{year+1}-01-01'

    image = (collection
             .filterDate(date_start, date_end)
             .filterBounds(geometry)
             .mosaic()
             .clip(geometry))

    return image


def prepare_embedding_export_image(image):
    """Prepare embedding image for export.

    Select all 64 embedding bands (A00-A63).
    (Adapted from google_embedding.py)

    Returns:
        ee.Image with 64 bands
    """
    band_names = [f'A{i:02d}' for i in range(64)]
    return image.select(band_names)


def download_embeddings_for_region(region_name, geometry, year, output_dir, scale=10):
    """Download Google embedding data for a specific region.

    Args:
        region_name: Name of the region (for filename)
        geometry: ee.Geometry for the region
        year: Year to download (2017-2024)
        output_dir: Output directory path
        scale: Resolution in meters (default 10m)

    Returns:
        Dict with download info or None if failed
    """
    print(f"  Querying Google embeddings for {region_name}...")

    # Query embedding collection
    try:
        image = query_embedding_for_region(geometry, year)

        # Check if data exists for this year/region
        collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        tile_count = (collection
                      .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                      .filterBounds(geometry)
                      .size()
                      .getInfo())

        if tile_count == 0:
            print(f"    No embeddings found for {year}")
            return None

        print(f"    Found {tile_count} tile(s) for {year}, creating mosaic...")

    except Exception as e:
        print(f"    Error querying embeddings: {e}")
        return None

    # Prepare for export
    print(f"    Preparing image (selecting 64 embedding bands)...")
    export_image = prepare_embedding_export_image(image)

    # Create filename
    sanitized_name = sanitize_region_name(region_name)
    filename = os.path.join(output_dir, f"google_embedding_{sanitized_name}_{year}.tif")

    print(f"    Downloading to: {filename}")
    print(f"    (This may take several minutes...)")

    # Download
    try:
        geemap.download_ee_image(
            export_image,
            filename=filename,
            region=geometry,
            scale=scale,
            crs='EPSG:4326'
        )

        # Get file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"    Done! Size: {file_size_mb:.1f} MB")

        return {
            'type': 'embedding',
            'region': region_name,
            'filename': os.path.basename(filename),
            'year': year,
            'size_mb': file_size_mb
        }

    except Exception as e:
        print(f"    Download error: {e}")
        return None


def write_summary_file(output_dir, downloads):
    """Write summary file with all download information.

    Args:
        output_dir: Output directory
        downloads: List of download info dicts
    """
    if not downloads:
        return

    summary_path = os.path.join(output_dir, 'kml_download_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KML Region Download Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Downloads: {len(downloads)}\n\n")

        # Group by type
        savi_downloads = [d for d in downloads if d['type'] == 'savi']
        embedding_downloads = [d for d in downloads if d['type'] == 'embedding']

        if savi_downloads:
            f.write("-" * 80 + "\n")
            f.write("SAVI Downloads:\n")
            f.write("-" * 80 + "\n")
            for d in savi_downloads:
                f.write(f"Region: {d['region']}\n")
                f.write(f"  File: {d['filename']}\n")
                f.write(f"  Date: {d['date']}\n")
                f.write(f"  Cloud: {d['cloud_pct']:.1f}%\n")
                f.write(f"  Size: {d['size_mb']:.1f} MB\n\n")

        if embedding_downloads:
            f.write("-" * 80 + "\n")
            f.write("Embedding Downloads:\n")
            f.write("-" * 80 + "\n")
            for d in embedding_downloads:
                f.write(f"Region: {d['region']}\n")
                f.write(f"  File: {d['filename']}\n")
                f.write(f"  Year: {d['year']}\n")
                f.write(f"  Size: {d['size_mb']:.1f} MB\n\n")

    print(f"\nSummary written to: {summary_path}")


def main():
    """Main execution workflow."""
    parser = argparse.ArgumentParser(
        description='Download GEE data for polygon regions defined in KML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download both SAVI and embeddings
  python kml_region_downloader.py --kml data/beer_sheva/sub_regions.kml \\
      --savi --year 2022 --embeddings --embedding-year 2022 \\
      --output data/beer_sheva/

  # Download only SAVI
  python kml_region_downloader.py --kml data/beer_sheva/sub_regions.kml \\
      --savi --year 2022 --output data/beer_sheva/

  # Download only embeddings
  python kml_region_downloader.py --kml data/beer_sheva/sub_regions.kml \\
      --embeddings --embedding-year 2022 --output data/beer_sheva/
        '''
    )

    parser.add_argument('--kml', type=str, required=True,
                        help='Path to KML file with polygon regions')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')

    # SAVI options
    parser.add_argument('--savi', action='store_true',
                        help='Download SAVI data')
    parser.add_argument('--year', '-y', type=int,
                        help='Year for SAVI download (2021-2024)')
    parser.add_argument('--savi-scale', type=int, default=30,
                        help='SAVI resolution in meters (default: 30)')

    # Embedding options
    parser.add_argument('--embeddings', action='store_true',
                        help='Download embedding data')
    parser.add_argument('--embedding-year', type=int,
                        help='Year for embedding download (2017-2024)')
    parser.add_argument('--embedding-scale', type=int, default=10,
                        help='Embedding resolution in meters (default: 10)')

    args = parser.parse_args()

    # Validation
    if not args.savi and not args.embeddings:
        parser.error("Must specify at least one of --savi or --embeddings")

    if args.savi and not args.year:
        parser.error("--year required when using --savi")

    if args.embeddings and not args.embedding_year:
        parser.error("--embedding-year required when using --embeddings")

    if not os.path.exists(args.kml):
        parser.error(f"KML file not found: {args.kml}")

    print("=" * 70)
    print("KML Region Downloader for Google Earth Engine")
    print("=" * 70)

    # Initialize GEE
    print("\nInitializing Google Earth Engine...")
    if not initialize_gee():
        return

    # Read KML regions
    print(f"\nReading KML file: {args.kml}")
    regions = read_kml_regions(args.kml)

    if not regions:
        print("No valid regions found in KML. Exiting.")
        return

    print(f"Found {len(regions)} region(s):")
    for name, geom in regions:
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        print(f"  - {name}: [{bounds[0]:.4f}, {bounds[1]:.4f}] to [{bounds[2]:.4f}, {bounds[3]:.4f}]")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process each region
    downloads = []

    for i, (region_name, shapely_geom) in enumerate(regions, 1):
        print(f"\n{'='*70}")
        print(f"Processing region {i}/{len(regions)}: {region_name}")
        print(f"{'='*70}")

        # Convert to EE geometry
        try:
            ee_geometry = shapely_to_ee_geometry(shapely_geom)
        except Exception as e:
            print(f"  Error converting geometry: {e}")
            print(f"  Skipping region: {region_name}")
            continue

        # Download SAVI if requested
        if args.savi:
            print(f"\n[SAVI Download]")
            result = download_savi_for_region(
                region_name,
                ee_geometry,
                args.year,
                args.output,
                scale=args.savi_scale
            )
            if result:
                downloads.append(result)

        # Download embeddings if requested
        if args.embeddings:
            print(f"\n[Embedding Download]")
            result = download_embeddings_for_region(
                region_name,
                ee_geometry,
                args.embedding_year,
                args.output,
                scale=args.embedding_scale
            )
            if result:
                downloads.append(result)

    # Write summary
    write_summary_file(args.output, downloads)

    # Final summary
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"Total downloads: {len(downloads)}")
    print(f"Output directory: {args.output}")

    if downloads:
        print("\nDownloaded files:")
        for d in downloads:
            print(f"  - {d['filename']}")


if __name__ == '__main__':
    main()
