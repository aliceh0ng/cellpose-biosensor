"""
src/io.py — loading and saving images and masks
"""
import numpy as np
import tifffile
from pathlib import Path


def load_stack(path: str | Path) -> np.ndarray:
    """
    Load a 16-bit 4-channel Zeiss Airyscan stack.

    Expected input: (4, H, W) uint16 TIFF
    Returns: (4, H, W) uint16 numpy array

    Channels by convention:
        [0] BFP  — constitutive (segmentation channel)
        [1] GFP  — osmotic stress reporter
        [2] RFP  — oxidative stress reporter
        [3] SYTOX — host nuclear stain (exclusion channel)
    """
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] == 4:
        pass  # already (C, H, W)
    elif img.ndim == 4:
        # (Z, C, H, W) — take first Z slice or max-project
        img = img.max(axis=0)
    else:
        raise ValueError(f"Unexpected image shape {img.shape} in {path}")

    if img.dtype != np.uint16:
        raise TypeError(f"Expected uint16, got {img.dtype} in {path}. "
                        "Use the 16-bit export from Zen, not the RGB preview.")
    return img


def get_bfp(stack: np.ndarray) -> np.ndarray:
    """Return the BFP channel (channel 0) as a 2D uint16 array."""
    return stack[0]


def save_masks(masks: np.ndarray, path: str | Path) -> None:
    """Save integer label mask as 16-bit TIFF. Each cell = unique integer ID."""
    tifffile.imwrite(str(path), masks.astype(np.uint16),
                     compression='lzw', photometric='minisblack')


def tile_image(img: np.ndarray, tile_size: int = 512,
               overlap: int = 64) -> list[dict]:
    """
    Split a 2D image into overlapping tiles for model inference.

    Returns a list of dicts with keys:
        'tile'  : (tile_size, tile_size) crop
        'y0', 'x0' : top-left corner in original image
    """
    H, W = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            # Pad if tile is smaller than tile_size
            tile = np.zeros((tile_size, tile_size), dtype=img.dtype)
            crop = img[y0:y1, x0:x1]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            tiles.append({'tile': tile, 'y0': y0, 'x0': x0,
                          'h': y1 - y0, 'w': x1 - x0})
    return tiles


def stitch_masks(tiles: list[dict], H: int, W: int,
                 overlap: int = 64) -> np.ndarray:
    """
    Reassemble tiled masks into a full-image label array.
    Uses simple max-value voting in overlap regions.
    For production use, consider the cellpose built-in stitch_threshold instead.
    """
    full = np.zeros((H, W), dtype=np.int32)
    offset = 0  # track global cell ID across tiles
    for t in tiles:
        y0, x0 = t['y0'], t['x0']
        h, w = t['h'], t['w']
        m = t['mask'][:h, :w].astype(np.int32)
        # Remap IDs to global space
        m_shifted = np.where(m > 0, m + offset, 0)
        offset += m.max()
        # Write into output (non-overlapping core region)
        pad = overlap // 2
        ys = max(0, y0 + pad)
        xs = max(0, x0 + pad)
        ye = min(H, y0 + h - pad)
        xe = min(W, x0 + w - pad)
        full[ys:ye, xs:xe] = m_shifted[ys-y0:ye-y0, xs-x0:xe-x0]
    return full
