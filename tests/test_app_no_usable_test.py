"""The dashboard must not crash when no test gives a usable result.

A heart recording that fails the quality check, with the X-ray left out of the
visit, leaves no usable test, and core.contracts.VisitDecision refuses to be built
without one. The page should say so instead of raising (which Streamlit shows as
an error box).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

os.environ["CARDIONARY_STAGE_HOLD"] = "0"
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
    at = testing.AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300).run()
    at.toggle(key="inc_xray").set_value(False).run()
    at.radio(key="src_heart_en").set_value("upload").run()
    at.file_uploader(key="up_heart").upload("noise.wav", buf.getvalue(), "audio/wav").run()   # analysed on upload
    at.button(key="btn_visit").click().run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "No visit decision" in text
    assert any("withheld" in w.value for w in at.warning)
    assert at.session_state["results"]["heart"]["withheld"] == "quality_check"
    assert 'class="ab withheld' in text                 # the heart input block shows it was withheld
