"""The dashboard must not crash when no test gives a usable result.

A heart recording that fails the quality check, with no X-ray, leaves no usable
test, and core.contracts.VisitDecision refuses to be built without one. The page
should say so instead of raising (which Streamlit shows as an error box).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = [ROOT / "models" / "chiron_cardio_v1_float.keras", ROOT / "models" / "chiron_cardio_v1_int8.tflite"]

testing = pytest.importorskip("streamlit.testing.v1")
element_tree = pytest.importorskip("streamlit.testing.v1.element_tree")


@pytest.mark.skipif(not all(p.exists() for p in MODELS), reason="model files not in this checkout")
@pytest.mark.skipif(not hasattr(getattr(element_tree, "FileUploader", None), "upload"),
                    reason="this Streamlit version cannot upload files in AppTest")
def test_withheld_heart_result_without_xray_shows_a_message_not_an_error():
    buf = io.BytesIO()
    noise = np.random.default_rng(0).normal(0, 0.3, 6 * 48000)      # fails the quality check
    wavfile.write(buf, 48000, (np.clip(noise, -1, 1) * 32767).astype(np.int16))
    at = testing.AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    at.file_uploader[0].upload("noise.wav", buf.getvalue(), "audio/wav").run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("No visit decision" in i.value for i in at.info)
    assert any("withheld" in w.value for w in at.warning)
