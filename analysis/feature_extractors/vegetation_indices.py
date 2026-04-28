#!/usr/bin/env python3
"""
Vegetation indices extractor for Landsat imagery.

Computes 6 vegetation and land cover indices from Landsat 8/9 GeoTIFF files.
"""

import numpy as np
import rasterio


# Index names and their purposes
VEGETATION_INDEX_NAMES = [
    'NDVI',   # Normalized Difference Vegetation Index
    'SAVI',   # Soil-Adjusted Vegetation Index
    'EVI',    # Enhanced Vegetation Index
    'NDWI',   # Normalized Difference Water Index
    'BSI',    # Bare Soil Index
    'NDBI',   # Normalized Difference Built-up Index
]


def extract_vegetation_indices(geotiff_path: str, savi_l: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """
    Compute 6 vegetation/land cover indices from a Landsat GeoTIFF.

    Landsat 8/9 band mapping:
        Band 2: Blue (0.45-0.51 μm)
        Band 4: Red (0.64-0.67 μm)
        Band 5: NIR (0.85-0.88 μm)
        Band 6: SWIR1 (1.57-1.65 μm)

    Indices computed:
        NDVI: (NIR - Red) / (NIR + Red)
        SAVI: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        NDWI: (NIR - SWIR1) / (NIR + SWIR1)
        BSI: ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)

    Args:
        geotiff_path: Path to GeoTIFF file with at least 6 bands
        savi_l: Soil brightness correction factor for SAVI (default 0.5)

    Returns:
        features: (H, W, 6) float32 array, each index normalized to [0, 1]
        names: List of index names

    Raises:
        ValueError: If file has fewer than 6 bands
    """
    with rasterio.open(geotiff_path) as src:
        if src.count < 6:
            raise ValueError(f"GeoTIFF has only {src.count} bands, need at least 6")

        # Read required bands (1-indexed in rasterio)
        blue = src.read(2).astype(np.float32)
        red = src.read(4).astype(np.float32)
        nir = src.read(5).astype(np.float32)
        swir1 = src.read(6).astype(np.float32)

    # Handle NaN/Inf values
    blue = np.nan_to_num(blue, nan=0, posinf=0, neginf=0)
    red = np.nan_to_num(red, nan=0, posinf=0, neginf=0)
    nir = np.nan_to_num(nir, nan=0, posinf=0, neginf=0)
    swir1 = np.nan_to_num(swir1, nan=0, posinf=0, neginf=0)

    # Compute indices
    ndvi = compute_ndvi(red, nir)
    savi = compute_savi(red, nir, savi_l)
    evi = compute_evi(blue, red, nir)
    ndwi = compute_ndwi(nir, swir1)
    bsi = compute_bsi(blue, red, nir, swir1)
    ndbi = compute_ndbi(nir, swir1)

    # Stack to (H, W, 6)
    features = np.stack([ndvi, savi, evi, ndwi, bsi, ndbi], axis=-1)

    # Normalize each index to [0, 1] using 2-98 percentile
    features = normalize_indices(features)

    return features, VEGETATION_INDEX_NAMES.copy()


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1], higher values indicate more vegetation
    """
    denominator = nir + red
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)
    return np.clip(ndvi, -1, 1)


def compute_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Compute SAVI (Soil-Adjusted Vegetation Index).

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    Better for sparse vegetation/desert regions, reduces soil brightness effects.
    L=0.5 is standard for most vegetation densities.
    Range: approximately [-1.5, 1.5]
    """
    denominator = nir + red + L
    with np.errstate(divide='ignore', invalid='ignore'):
        savi = np.where(denominator > 0, ((nir - red) / denominator) * (1 + L), 0)
    return np.clip(savi, -1.5, 1.5)


def compute_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute EVI (Enhanced Vegetation Index).

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    Reduces atmospheric and soil background influences.
    Better for high biomass regions than NDVI.
    Range: approximately [-1, 1]
    """
    denominator = nir + 6 * red - 7.5 * blue + 1
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = np.where(denominator > 0, 2.5 * (nir - red) / denominator, 0)
    return np.clip(evi, -1, 1)


def compute_ndwi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute NDWI (Normalized Difference Water Index).

    NDWI = (NIR - SWIR1) / (NIR + SWIR1)
    Indicates vegetation water content / moisture stress.
    Higher values = more water content.
    Range: [-1, 1]
    """
    denominator = nir + swir1
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(denominator > 0, (nir - swir1) / denominator, 0)
    return np.clip(ndwi, -1, 1)


def compute_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute BSI (Bare Soil Index).

    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    Higher values indicate more bare soil exposure.
    Range: [-1, 1]
    """
    numerator = (swir1 + red) - (nir + blue)
    denominator = (swir1 + red) + (nir + blue)
    with np.errstate(divide='ignore', invalid='ignore'):
        bsi = np.where(denominator > 0, numerator / denominator, 0)
    return np.clip(bsi, -1, 1)


def compute_ndbi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute NDBI (Normalized Difference Built-up Index).

    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Higher values indicate built-up/urban areas.
    Note: This is the inverse of NDWI.
    Range: [-1, 1]
    """
    denominator = swir1 + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndbi = np.where(denominator > 0, (swir1 - nir) / denominator, 0)
    return np.clip(ndbi, -1, 1)


def normalize_indices(features: np.ndarray) -> np.ndarray:
    """
    Normalize each index to [0, 1] using 2-98 percentile clipping.

    This shifts and scales the data so that:
    - 2nd percentile maps to 0
    - 98th percentile maps to 1

    Args:
        features: (H, W, n_indices) array

    Returns:
        Normalized array with values in [0, 1]
    """
    normalized = np.zeros_like(features, dtype=np.float32)

    for i in range(features.shape[-1]):
        band = features[:, :, i]
        # Use all values for indices (they can be negative)
        valid = band[np.isfinite(band)]

        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                # Constant value - set to 0.5
                normalized[:, :, i] = 0.5

    return normalized


def get_index_descriptions() -> dict[str, str]:
    """
    Get descriptions for each vegetation index.

    Returns:
        Dictionary mapping index name to description
    """
    return {
        'NDVI': 'Normalized Difference Vegetation Index - measures vegetation greenness',
        'SAVI': 'Soil-Adjusted Vegetation Index - better for sparse vegetation/desert',
        'EVI': 'Enhanced Vegetation Index - reduces atmospheric effects',
        'NDWI': 'Normalized Difference Water Index - vegetation water content',
        'BSI': 'Bare Soil Index - indicates bare soil exposure',
        'NDBI': 'Normalized Difference Built-up Index - identifies urban areas',
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vegetation_indices.py <geotiff_path>")
        sys.exit(1)

    path = sys.argv[1]
    features, names = extract_vegetation_indices(path)
    print(f"Extracted indices shape: {features.shape}")
    print(f"Index names: {names}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Print mean values for each index
    for i, name in enumerate(names):
        print(f"  {name}: mean={features[:,:,i].mean():.3f}")
