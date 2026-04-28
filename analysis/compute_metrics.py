#!/usr/bin/env python3
"""Compute all metrics for the paper table."""

import numpy as np
from skimage import measure, filters, morphology
import rasterio


def compute_curvature_variance(mask):
    """Compute curvature variance of the longest contour."""
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    longest = max(contours, key=len)
    x, y = longest[:, 1], longest[:, 0]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)**1.5
    return np.var(curvature)


def compute_metrics(region='beer_sheva'):
    """Compute all metrics for a region."""

    if region == 'beer_sheva':
        boundary_path = 'data/beer_sheva/boundary/boundary_embedding.npz'
        ndvi_path = 'data/beer_sheva/LC08_20221001_SAVI.tif'
        name = 'Israel (Beer Sheva)'
    else:
        boundary_path = 'data/algeria/boundary/boundary_embedding.npz'
        ndvi_path = 'data/algeria/LC08_algeria_2023_composite_SAVI.tif'
        name = 'Algeria'

    # Load boundary data (Ours - embedding method)
    data = np.load(boundary_path, allow_pickle=True)
    fit_stats = data['fit_stats'].item()
    smoothed_mask_ours = data['smoothed_mask'].astype(bool)
    threshold_used = data['threshold'][0]

    print(f"=== {name} ===")
    print(f"Threshold used: {threshold_used:.3f}")
    print()
    print(f"R² (embed → NDVI): {fit_stats['r2']:.2f}")
    print(f"Correlation: {fit_stats['correlation']:.2f}")

    # Curvature for Ours
    curv_ours = compute_curvature_variance(smoothed_mask_ours)
    print(f"Curvature variance (Ours): {curv_ours:.3f}")

    # Load and process NDVI
    with rasterio.open(ndvi_path) as src:
        red = src.read(4).astype(np.float32)
        nir = src.read(5).astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = 0
    valid_mask = (red != 0) | (nir != 0)

    # Use Otsu for NDVI
    valid_values = ndvi[valid_mask]
    vmin, vmax = valid_values.min(), valid_values.max()
    normalized = (valid_values - vmin) / (vmax - vmin + 1e-10)
    otsu_norm = filters.threshold_otsu(normalized)
    threshold_ndvi = otsu_norm * (vmax - vmin) + vmin

    ndvi_binary = (ndvi > threshold_ndvi) & valid_mask

    # Apply same smoothing as Ours (kernel=20, iter=7)
    kernel = morphology.disk(20)
    ndvi_smooth = ndvi_binary.astype(np.uint8)
    for _ in range(7):
        ndvi_smooth = morphology.closing(ndvi_smooth, kernel)
        ndvi_smooth = morphology.opening(ndvi_smooth, kernel)
    ndvi_smooth = ndvi_smooth.astype(bool)

    # Curvature for NDVI
    curv_ndvi = compute_curvature_variance(ndvi_smooth)
    print(f"Curvature variance (NDVI): {curv_ndvi:.3f}")

    if curv_ours and curv_ndvi:
        smoothness_gain = (curv_ndvi - curv_ours) / curv_ndvi * 100
        print(f"Smoothness gain: {smoothness_gain:.1f}%")

    print()
    return {
        'r2': fit_stats['r2'],
        'correlation': fit_stats['correlation'],
        'curv_ours': curv_ours,
        'curv_ndvi': curv_ndvi,
    }


if __name__ == '__main__':
    print("Computing metrics for paper table...\n")

    israel = compute_metrics('beer_sheva')
    algeria = compute_metrics('algeria')

    print("\n" + "="*50)
    print("LaTeX Table Values:")
    print("="*50)
    print(f"R² Israel: {israel['r2']:.2f}")
    print(f"R² Algeria: {algeria['r2']:.2f}")
    print(f"Correlation Israel: {israel['correlation']:.2f}")
    print(f"Correlation Algeria: {algeria['correlation']:.2f}")
    print(f"Curvature Ours Israel: {israel['curv_ours']:.3f}")
    print(f"Curvature Ours Algeria: {algeria['curv_ours']:.3f}")
    print(f"Curvature NDVI Israel: {israel['curv_ndvi']:.3f}")
    print(f"Curvature NDVI Algeria: {algeria['curv_ndvi']:.3f}")
