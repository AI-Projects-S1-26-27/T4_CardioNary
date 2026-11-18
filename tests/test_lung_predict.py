"""Unit tests for the X-ray prediction interface without model artifacts."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from core.contracts import PatientMetadata
from lung import predict
from lung.fusion import late_fusion


def _png_bytes() -> bytes:
    pixels = np.tile(np.arange(120, dtype=np.uint8), (80, 1))
    image = Image.fromarray(pixels, mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_load_image_from_bytes_center_crops_and_resizes():
    image = predict._load_img_from_source(_png_bytes())

    assert image.shape == (96, 96)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_display_path_uses_shared_features_and_clahe(monkeypatch):
    from lung.model import clahe, features

    image = np.zeros((96, 96), dtype=np.float32)
    expected_features = {"clahe_mean": 1.0, "clahe_std": 2.0, "clahe_abs_laplacian": 3.0}
    expected_view = np.full((96, 96), 7, dtype=np.uint8)
    monkeypatch.setattr(features, "characteristics", lambda _image: expected_features)
    monkeypatch.setattr(clahe, "clahe_view", lambda _image: expected_view)

    assert predict._safe_image_features(image) == expected_features
    assert predict._safe_clahe_view(image) is expected_view


def test_analyse_xray_returns_calibrated_cnn_clahe_and_occlusion(monkeypatch):
    monkeypatch.setattr(predict, "_get_norm", lambda: (0.0, 1.0))
    monkeypatch.setattr(predict, "_get_interp", lambda: object())
    monkeypatch.setattr(predict, "_temperature", lambda: 1.0)
    monkeypatch.setattr(predict, "_safe_image_features", lambda image: {"mean_intensity": 0.5})
    monkeypatch.setattr(predict, "_safe_clahe_view", lambda image: np.zeros((96, 96), dtype=np.uint8))
    calls = []

    def predict_batch(_interp, images, _mu, _sd):
        calls.append(images.copy())
        return np.full(len(images), 0.8 if len(images) == 1 else 0.5)

    monkeypatch.setattr(predict, "_predict_int8", predict_batch)
    # The fitted fusion callable is supplied by the feature-vote owner. Keep
    # this unit test deterministic while that callable is not available.
    monkeypatch.setattr(late_fusion, "xray_fusion", None, raising=False)

    model_out, fused_out, clahe, occlusion = predict.analyse_xray(
        _png_bytes(), PatientMetadata(age_years=57, sex="female")
    )

    assert model_out.probabilities["Pneumonia"] == 0.8
    assert fused_out.p_abnormal == 0.8
    assert fused_out.votes == {"cnn": 0.8}
    assert fused_out.weights == {"cnn": 1.0}
    assert clahe.shape == (96, 96)
    assert occlusion.shape == (3, 3)
    assert np.allclose(occlusion, 0.3)
    assert [len(batch) for batch in calls] == [1, 9]
