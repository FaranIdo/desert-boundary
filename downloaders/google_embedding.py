#!/usr/bin/env python3
"""
Google Satellite Embedding Downloader

Downloads Google Satellite Embedding V1 Annual dataset from Google Earth Engine
for the Beer Sheva study area.

The embeddings are 64-dimensional learned representations at 10m resolution
that encode temporal trajectories of surface conditions.

Features:
- Query by year (2017-2024)
- Download all 64 embedding bands (A00-A63)
- Same study area as Landsat downloads

Usage:
    python google_embedding.py --year 2022
    python google_embedding.py --year 2022 --scale 10
"""

import ee
import geemap
import argparse
import os

# Initialize Earth Engine with project (required for authentication)
GEE_PROJECT_ID = 'ee-faranido'

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
        [3.676376308297045, 36.17830529911284],
        [2.435886377181717, 36.17079370431332],
        [2.450111959364443, 35.49361267330211],
        [3.681691218173533, 35.51074295395308]
    ]
}


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


def query_embedding_image(year, geometry):
    """Query Google Satellite Embedding V1 Annual image for a specific year.

    Args:
        year: Year to query (2017-2024)
        geometry: Study area geometry

    Returns:
        ee.Image for the specified year
    """
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Filter to specific year using proper date range (inclusive of full year)
    date_start = f'{year}-01-01'
    date_end = f'{year+1}-01-01'

    image = (collection
             .filterDate(date_start, date_end)
             .filterBounds(geometry)
             .mosaic()
             .clip(geometry))

    return image


def prepare_export_image(image, geometry):
    """Prepare embedding image for export.

    Selects all 64 embedding bands (A00-A63). Image is already clipped
    to study area in query_embedding_image().

    Args:
        image: ee.Image with embedding bands (already clipped)
        geometry: Study area geometry (unused, kept for API compatibility)

    Returns:
        ee.Image ready for export
    """
    # Select all 64 embedding bands
    band_names = [f'A{i:02d}' for i in range(64)]
    return image.select(band_names)


def download_image(image, filename, region, scale=10):
    """Download image to local GeoTIFF using geemap.

    Args:
        image: ee.Image to download
        filename: Output filename
        region: ee.Geometry for the region of interest
        scale: Resolution in meters (default 10m - native resolution)

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


def write_summary_file(output_dir, filename, year, region, file_size_mb, study_area_coords):
    """Write a summary file with metadata for the download.

    Args:
        output_dir: Directory where file is saved
        filename: Name of downloaded file
        year: Year of the embedding data
        region: Region name (beer_sheva or algeria)
        file_size_mb: Size of file in MB
        study_area_coords: The polygon coordinates used
    """
    from datetime import datetime

    summary_path = os.path.join(output_dir, f'google_embedding_{region}_{year}_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Google Satellite Embedding V1 Annual Download Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL\n")
        f.write(f"Year: {year}\n")
        f.write(f"Resolution: 10m (native)\n\n")

        f.write("Study Area Polygon:\n")
        for i, coord in enumerate(study_area_coords):
            f.write(f"  Point {i+1}: [{coord[0]}, {coord[1]}]\n")
        f.write("\n")

        f.write("Embedding Information:\n")
        f.write("  - 64 bands (A00-A63)\n")
        f.write("  - Each pixel is a 64-dimensional unit-length vector\n")
        f.write("  - Represents learned geospatial representations\n")
        f.write("  - Encodes temporal trajectories of surface conditions\n\n")

        f.write("Output Bands:\n")
        for i in range(64):
            f.write(f"  {i+1}. A{i:02d} - Embedding dimension {i}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("Downloaded File:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Filename: {filename}\n")
        f.write(f"Size: {file_size_mb:.1f} MB\n")
        f.write(f"Year: {year}\n")
        f.write("-" * 80 + "\n")

    return summary_path


def main():
    """Main execution workflow."""
    parser = argparse.ArgumentParser(
        description='Download Google Satellite Embedding from Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python google_embedding.py --year 2022           # Download 2022 embeddings
  python google_embedding.py --year 2022 --scale 20  # Download at 20m resolution
  python google_embedding.py --output ./output     # Custom output directory
        '''
    )
    parser.add_argument('--year', '-y', type=int, required=True,
                        help='Year to download (2017-2024)')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--scale', '-s', type=int, default=10,
                        help='Resolution in meters (default: 10m)')
    parser.add_argument('--region', '-r', type=str, default='beer_sheva',
                        choices=['beer_sheva', 'algeria'],
                        help='Study area region (default: beer_sheva)')

    args = parser.parse_args()

    # Validate year
    if args.year < 2017 or args.year > 2024:
        parser.error("Year must be between 2017 and 2024")

    print("=" * 60)
    print("Google Satellite Embedding Downloader")
    print("=" * 60)

    # Initialize
    print("\nInitializing Google Earth Engine...")
    initialize_gee()

    # Get study area
    geometry = get_study_area(args.region)
    study_area_coords = STUDY_AREAS[args.region]
    print(f"Region: {args.region}")
    print(f"Study area: {study_area_coords[0]} to {study_area_coords[2]}")

    # Query image for the year
    print(f"\nQuerying embeddings for year {args.year}...")
    image = query_embedding_image(args.year, geometry)

    # Check if tiles exist for this year
    try:
        collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        tile_count = (collection
                      .filterDate(f'{args.year}-01-01', f'{args.year+1}-01-01')
                      .filterBounds(geometry)
                      .size()
                      .getInfo())
        if tile_count == 0:
            print(f"No embeddings found for year {args.year}.")
            return
        print(f"Found {tile_count} tiles for {args.year}, creating mosaic...")
        print(f"Bands: 64 (A00-A63)")
    except Exception as e:
        print(f"Error checking embeddings: {e}")
        return

    # Prepare for export
    print("\nPreparing image (selecting all 64 embedding bands)...")
    export_image = prepare_export_image(image, geometry)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Download
    filename = os.path.join(args.output, f"google_embedding_{args.region}_{args.year}.tif")
    print(f"\nDownloading to: {filename}")
    print(f"Resolution: {args.scale}m")
    print(f"(This may take several minutes for large regions...)")

    success = download_image(export_image, filename, geometry, scale=args.scale)

    if success:
        print(f"Done!")

        # Get file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")

        # Write summary
        print("\nWriting summary file...")
        summary_path = write_summary_file(
            args.output,
            os.path.basename(filename),
            args.year,
            args.region,
            file_size_mb,
            study_area_coords
        )
        print(f"Summary: {summary_path}")
    else:
        print(f"Failed to download. Try with --scale 20 for faster download.")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
