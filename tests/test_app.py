"""Tests for the CardioNary Streamlit Clinical Dashboard (app/app.py).

Verifies:
- Clean offline rendering without exceptions.
- English and Hindi (हिंदी) language toggling.
- Strict compliance with the 'NO RED' clinical rule (no st.error).
- Execution of multimodal inference and valid VisitDecision findings.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "app.py"


def test_app_loads_without_exceptions():
    """Initial load must render cleanly with zero errors and no red alerts."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception
    # Strict NO RED rule: no st.error elements
    assert len(at.error) == 0


def test_app_hindi_toggle():
    """Toggling to Hindi should update titles and labels dynamically."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    lang_radio = at.sidebar.radio[0]
    lang_radio.set_value("हिंदी").run()
    assert not at.exception
    assert len(at.error) == 0
    combined_text = " ".join([m.value for m in at.markdown])
    assert "कार्डियोनेरी" in combined_text or "स्क्रीनिंग" in combined_text


def test_app_sample_inference():
    """Selecting a test audio sample and running analysis produces clinical findings."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    # Select the first sample from the Heart sound dropdown
    sb_heart = at.sidebar.selectbox[1]
    if len(sb_heart.options) > 1:
        sb_heart.select(sb_heart.options[1]).run()
        main_buttons = [b for b in at.button if "Run" in b.label or "स्क्रीनिंग" in b.label]
        if main_buttons:
            main_buttons[0].click().run()
            assert not at.exception
            # Verify NO RED error boxes
            assert len(at.error) == 0
            # Verify findings rendered in markdown
            combined_text = " ".join([m.value for m in at.markdown])
            assert "Findings" in combined_text or "परिणाम" in combined_text
            assert "Routine" in combined_text or "Referral" in combined_text or "जोखिम" in combined_text or "परामर्श" in combined_text
