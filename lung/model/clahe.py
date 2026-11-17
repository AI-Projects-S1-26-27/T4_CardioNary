"""CLAHE view of a 96 x 96 chest X-ray and the three texture numbers taken from it.

Roadmap S1 (Harsh), proposal 4.2. OpenCV CLAHE, clip limit 2.0, 8 x 8 tiles.
The view is shown in the dashboard. Over the lung area it gives:

    clahe_mean           mean of the enhanced pixels (0 to 255)
    clahe_std            standard deviation of the enhanced pixels
    clahe_abs_laplacian  mean absolute Laplacian of the enhanced image (edge/texture energy)

The lung area is the central thorax box without the mediastinum strip, the same
region that core.descriptors uses for the dense-area measure.
"""

from __future__ import annotations

import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)
CLAHE_COLUMNS = ("clahe_mean", "clahe_std", "clahe_abs_laplacian")


def lung_area(h: int = 96, w: int = 96) -> np.ndarray:
    """Boolean mask: image minus an 8% border and minus the central 12% column strip."""
    b = int(0.08 * h)
    roi = np.zeros((h, w), dtype=bool)
    roi[b:h - b, b:w - b] = True
    roi[:, int(0.44 * w):int(0.56 * w)] = False
    return roi


def to_uint8(img01: np.ndarray) -> np.ndarray:
    """Image in [0, 1] -> uint8, clipping anything outside the range."""
    return np.clip(np.nan_to_num(np.asarray(img01, dtype=np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def clahe_view(img01: np.ndarray) -> np.ndarray:
    """The contrast-enhanced view of a 2-D image in [0, 1], as uint8 (for display)."""
    img01 = np.asarray(img01)
    if img01.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {img01.shape}")
    return cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID).apply(to_uint8(img01))


def clahe_numbers(img01: np.ndarray, view: np.ndarray | None = None) -> dict[str, float]:
    """The three texture numbers for the lung area of the CLAHE view."""
    view = clahe_view(img01) if view is None else view
    roi = lung_area(*view.shape)
    lap = cv2.Laplacian(view.astype(np.float32), cv2.CV_32F)
    return {
        "clahe_mean": float(view[roi].mean()),
        "clahe_std": float(view[roi].std()),
        "clahe_abs_laplacian": float(np.abs(lap[roi]).mean()),
    }
