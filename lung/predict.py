"""Chest X-ray prediction interface for the dashboard.

One call takes a chest radiograph and the patient's metadata and returns
core.contracts outputs, a CLAHE view, and a 3x3 occlusion map.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from core.contracts import FusedOutput, ModelOutput, PatientMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_JSON = REPO_ROOT / "reports" / "xray_calibration_kermany.json"

_INTERP = None
_NORM = None


def _get_interp():
    global _INTERP
    if _INTERP is None:
        from lung.eval.common import load_int8
        _INTERP = load_int8()
    return _INTERP


def _get_norm():
    global _NORM
    if _NORM is None:
        from lung.eval.common import load_norm
        _NORM = load_norm()
    return _NORM


def _predict_int8(interp, img01_batch, mu, sd):
    from lung.eval.common import predict_int8
    return predict_int8(interp, img01_batch, mu, sd)


def _safe_clahe_view(img01: np.ndarray) -> np.ndarray:
    """Contrast-enhanced view for display, with pure-numpy fallback if cv2 is absent."""
    try:
        import cv2
        u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    except ImportError:
        lo, hi = np.percentile(img01, [2, 98])
        scaled = np.clip((img01 - lo) / (hi - lo), 0, 1) if hi > lo else img01
        return (scaled * 255.0).astype(np.uint8)


def _safe_image_features(img01: np.ndarray) -> dict:
    """Secondary texture features using core.descriptors with cv2 optional enhancement."""
    from core.descriptors import vision_descriptors
    desc = vision_descriptors(img01)
    zones = desc.get("zone_distribution_pct", {})
    row = {
        "dense_area_pct": desc["dense_area_pct"],
        "asymmetry_index": desc["asymmetry_index"],
        "zone_upper_pct": zones.get("upper", 0.0),
        "zone_middle_pct": zones.get("middle", 0.0),
        "zone_lower_pct": zones.get("lower", 0.0),
        "largest_region_fraction": desc["largest_region_fraction"],
        "regions_detected": desc["regions_detected"],
        "mean_intensity": desc["mean_intensity"],
        "texture_heterogeneity": desc["texture_heterogeneity"],
    }
    try:
        import cv2
        h, w = img01.shape
        b = int(0.08 * h)
        roi = np.zeros((h, w), dtype=bool)
        roi[b:h - b, b:w - b] = True
        roi[:, int(0.44 * w):int(0.56 * w)] = False
        u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
        for name, clip in (("clahe_std_1", 1.0), ("clahe_std_2", 2.0), ("clahe_std_4", 4.0)):
            f = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            row[name] = round(float(np.std(f.apply(u8)[roi])), 1)
    except ImportError:
        pass
    return row


def _calibrate_probability(probability: float, temperature: float) -> float:
    """Apply binary temperature scaling to a probability."""
    probability = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    logit = math.log(probability / (1.0 - probability)) / temperature
    return float(1.0 / (1.0 + math.exp(-logit)))


def _temperature() -> float:
    if CALIBRATION_JSON.exists():
        try:
            data = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
            value = float(data.get("cnn_oof_kermany", {}).get("temperature_T", 0.9442))
            if math.isfinite(value) and value > 0:
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return 0.9442


def _occlusion_map_3x3(interp, img01: np.ndarray, mu: float, sd: float, temperature: float,
                       baseline_probability: float) -> np.ndarray:
    """Return calibrated probability drops after masking each of nine image regions."""
    h, w = img01.shape
    row_edges = np.linspace(0, h, 4, dtype=int)
    col_edges = np.linspace(0, w, 4, dtype=int)
    masked = np.repeat(img01[None, ...], 9, axis=0)
    fill_value = float(np.mean(img01))
    for index in range(9):
        row, col = divmod(index, 3)
        masked[index, row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]] = fill_value

    occluded = np.asarray(_predict_int8(interp, masked, mu, sd), dtype=float).reshape(-1)
    calibrated = np.asarray([_calibrate_probability(p, temperature) for p in occluded])
    return (baseline_probability - calibrated).reshape(3, 3).astype(np.float32)


def _fuse_xray(p_cnn: float, features: dict, metadata: Optional[PatientMetadata]) -> FusedOutput:
    """Use the fitted X-ray fusion when available; otherwise report only the CNN vote.

    The feature and metadata votes must come from Harsh's frozen parameter file.
    Never substitute guessed probabilities or weights when that artifact is absent.
    """
    from lung.fusion import late_fusion
    xray_fusion = getattr(late_fusion, "xray_fusion", None)
    if xray_fusion is None:
        return FusedOutput(
            modality="xray",
            p_abnormal=float(p_cnn),
            votes={"cnn": float(p_cnn)},
            weights={"cnn": 1.0},
        )
    return xray_fusion(p_cnn, features, metadata)


def _load_img_from_source(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    if isinstance(source, (str, Path)):
        im = Image.open(source)
    elif isinstance(source, bytes):
        im = Image.open(io.BytesIO(source))
    elif hasattr(source, "read"):
        im = Image.open(source)
    else:
        raise ValueError(f"Unsupported image input type: {type(source)}")

    with im:
        im = im.convert("L")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        im = im.resize((96, 96), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def analyse_xray(
    image: Union[str, Path, bytes, io.BytesIO],
    metadata: Optional[PatientMetadata] = None,
) -> tuple[ModelOutput, FusedOutput, np.ndarray, np.ndarray]:
    """Run X-ray inference; return model/fusion outputs, CLAHE and 3x3 occlusion map."""
    img01 = _load_img_from_source(image)
    mu, sd = _get_norm()
    interp = _get_interp()

    # Raw CNN prediction
    p_raw = float(_predict_int8(interp, img01[None, ...], mu, sd)[0])

    T = _temperature()
    p_cal = _calibrate_probability(p_raw, T)

    # Secondary characteristics & CLAHE
    sec = _safe_image_features(img01)
    clahe_img = _safe_clahe_view(img01)
    occlusion_map = _occlusion_map_3x3(interp, img01, mu, sd, T, p_cal)

    model_out = ModelOutput(
        modality="xray",
        probabilities={"Normal": float(1.0 - p_cal), "Pneumonia": float(p_cal)},
        secondary={k: float(v) for k, v in sec.items() if math.isfinite(float(v))},
        quality_ok=True,
    )

    fused_out = _fuse_xray(p_cal, sec, metadata)

    return model_out, fused_out, clahe_img, occlusion_map
