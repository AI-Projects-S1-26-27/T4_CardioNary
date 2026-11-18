"""Tests for the CardioNary dashboard (app/app.py), in the Chiron station's layout.

Checks:
- the page loads with no exception and no error box (no red: st.error is never used);
- English and Hindi, and that switching language keeps what was entered;
- one visit analysed through both branches gives a decision and lights the architecture;
- stepping to the next reference case analyses it (Chiron's case navigation);
- the screening batch grades every reference case.

The stages stay lit for CARDIONARY_STAGE_HOLD seconds; the tests set it to 0.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["CARDIONARY_STAGE_HOLD"] = "0"
testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"
MODELS = [ROOT / "models" / "chiron_cardio_v1_int8.tflite", ROOT / "models" / "chiron_pulmo_v1_int8.tflite"]
needs_models = pytest.mark.skipif(not all(p.exists() for p in MODELS), reason="model files not in this checkout")


def app():
    return testing.AppTest.from_file(str(APP_PATH), default_timeout=300).run()


def page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_loads_without_exceptions():
    at = app()
    assert not at.exception
    assert len(at.error) == 0
    text = page_text(at)
    assert "CardioNary" in text and "<svg" in text and "System architecture" in text
    assert "No visit analysed yet" in text


def test_app_hindi_toggle_keeps_the_patient_details():
    at = app()
    at.selectbox(key="m_sex_en").set_value("female").run()
    at.multiselect(key="m_symp_en").set_value(["cough"]).run()
    at.selectbox(key="lang").set_value("hi").run()
    assert not at.exception and len(at.error) == 0
    assert "हृदय-फेफड़ा जाँच केंद्र" in page_text(at)
    assert at.selectbox(key="m_sex_hi").value == "female"
    at.selectbox(key="lang").set_value("en").run()           # the round trip that once broke
    assert not at.exception and len(at.error) == 0
    assert at.session_state["meta_sex"] == "female" and at.session_state["meta_symp"] == ["cough"]
    assert at.selectbox(key="m_sex_en").value == "female"


@needs_models
def test_visit_analysis_gives_one_decision():
    at = app()
    at.button(key="btn_visit").click().run()
    assert not at.exception and len(at.error) == 0
    text = page_text(at)
    assert "Refer for clinical review" in text or "Routine follow-up" in text
    assert "Analysis complete" in text
    assert 'class="ab done' in text                     # finished blocks are marked done
    results = at.session_state["results"]
    assert results["heart"]["pred"] and results["xray"]["pred"]
    assert results["heart"]["meta_sig"] is None         # no patient details: no metadata vote made up
    assert "metadata" not in results["heart"]["votes"]
    assert len(at.session_state["records"]) == 2 and not at.session_state["pending"]


@needs_models
def test_next_case_is_analysed():
    at = app()
    at.button(key="next_heart").click().run()
    assert not at.exception and len(at.error) == 0
    assert at.session_state["idx_heart"] == 1
    assert at.session_state["results"]["heart"]["case"] == 1
    assert at.session_state["results"]["xray"] is None  # only the card that moved runs


@needs_models
def test_screening_batch_grades_every_reference_case():
    at = app()
    at.button(key="btn_batch").click().run()
    assert not at.exception and len(at.error) == 0
    records = at.session_state["records"]
    assert len(records) == 53 + 120 and all(r["batch"] for r in records)
    assert sum(1 for r in records if r["correct"]) >= 0.9 * len(records)
    assert "53 of 53 reference cases analysed" in page_text(at)
