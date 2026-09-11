"""
Preprocessing Module for Sentinel-2 Optical and Sentinel-1 SAR imagery.
Provides normalization, patch tiling, and scene reconstruction functions.
"""

import numpy as np
from typing import Tuple, List, Optional


def normalize_optical(data: np.ndarray, max_val: float = 10000.0) -> np.ndarray:
    """Normalize Sentinel-2 optical reflectance bands (L1C/L2A) to [0.0, 1.0].

    Args:
        data: Optical numpy array of shape (C, H, W) or (H, W, C).
        max_val: Reflectance scale factor (typically 10000.0).

    Returns:
        Normalized float32 array in [0.0, 1.0].
    """
    clipped = np.clip(data, 0.0, max_val)
    return (clipped / max_val).astype(np.float32)


def denormalize_optical(data: np.ndarray, max_val: float = 10000.0) -> np.ndarray:
    """Convert normalized [0.0, 1.0] optical tensor back to original reflectance scale.

    Args:
        data: Normalized optical array.
        max_val: Original reflectance scale factor.

    Returns:
        Reflectance values array.
    """
    return np.clip(data * max_val, 0.0, max_val).astype(np.float32)


def normalize_sar(
    data: np.ndarray,
    vv_min: float = -25.0,
    vv_max: float = 0.0,
    vh_min: float = -32.0,
    vh_max: float = -5.0
) -> np.ndarray:
    """Convert raw Sentinel-1 SAR linear power values to decibels (dB) and normalize to [0.0, 1.0].

    Args:
        data: SAR array of shape (2, H, W) where channel 0 is VV and channel 1 is VH.
        vv_min, vv_max: Clipping bounds in dB for VV channel.
        vh_min, vh_max: Clipping bounds in dB for VH channel.

    Returns:
        Normalized float32 SAR array of shape (2, H, W) in range [0.0, 1.0].
    """
    out = np.zeros_like(data, dtype=np.float32)
    
    # If inputs are already in dB or linear power:
    # Convert linear power to dB if min values are positive
    if np.min(data) >= 0.0:
        db_data = 10.0 * np.log10(np.maximum(data, 1e-5))
    else:
        db_data = data.copy()

    # VV Channel (0)
    vv_norm = (np.clip(db_data[0], vv_min, vv_max) - vv_min) / (vv_max - vv_min)
    out[0] = vv_norm

    # VH Channel (1)
    if data.shape[0] > 1:
        vh_norm = (np.clip(db_data[1], vh_min, vh_max) - vh_min) / (vh_max - vh_min)
        out[1] = vh_norm

    return out.astype(np.float32)


def tile_image(
    img: np.ndarray,
    tile_size: int = 256,
    stride: int = 256
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Tile a large (C, H, W) satellite scene into sub-patch tiles.

    Args:
        img: Input array of shape (C, H, W).
        tile_size: Height and width of each tile.
        stride: Stride between adjacent tiles.

    Returns:
        Tuple of (list_of_tiles, list_of_top_left_coords).
    """
    c, h, w = img.shape
    tiles = []
    coords = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[:, y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            coords.append((y, x))

    return tiles, coords


def reassemble_tiles(
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int]],
    orig_shape: Tuple[int, int, int],
    tile_size: int = 256
) -> np.ndarray:
    """Reassemble patch tiles into full scene output array.

    Args:
        tiles: List of (C, tile_size, tile_size) tiles.
        coords: List of (y, x) top-left coordinates.
        orig_shape: (C, H, W) shape of target full scene.
        tile_size: Patch size.

    Returns:
        Reconstructed scene of shape (C, H, W).
    """
    c, h, w = orig_shape
    output = np.zeros((c, h, w), dtype=np.float32)
    count = np.zeros((1, h, w), dtype=np.float32)

    for tile, (y, x) in zip(tiles, coords):
        output[:, y:y + tile_size, x:x + tile_size] += tile
        count[:, y:y + tile_size, x:x + tile_size] += 1.0

    count = np.maximum(count, 1.0)
    return output / count
