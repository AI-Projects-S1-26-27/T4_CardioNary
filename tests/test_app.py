"""Tests for the CardioNary dashboard (app/app.py), in the Chiron station's layout.

Checks:
- the page loads with no exception and no error box (no red: st.error is never used),
  and the first reference heart sound and X-ray are there before anything is run;
- the six languages, and that switching language keeps what was entered;
- the pregnancy field is greyed out, and not counted, for a male patient;
- the reference cases: diseased cases first, and every one answered correctly;
- one visit analysed through both branches gives a decision and lights the architecture;
- stepping to the next reference case analyses it (Chiron's case navigation);
- the screening batch grades every reference case.

The stages stay lit for CARDIONARY_STAGE_HOLD seconds; the tests set it to 0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["CARDIONARY_STAGE_HOLD"] = "0"
testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"
DEMO = json.loads((ROOT / "app" / "demo_cases.json").read_text(encoding="utf-8"))
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


def test_reference_inputs_are_shown_before_any_run():
    at = app()
    assert not at.exception
    assert len(at.get("audio")) == 1                           # the first reference heart sound, ready to play
    text = page_text(at)
    assert 'class="px" src="data:image/png;base64,' in text      # the first reference X-ray
    assert "Source radiograph" in text and "Source heart sound" in text
    assert at.session_state["results"] == {"heart": None, "xray": None} and not at.session_state["records"]


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


@pytest.mark.parametrize("lang, title", [("bn", "হৃদ্‌যন্ত্র-ফুসফুস স্ক্রিনিং কেন্দ্র"),
                                         ("te", "గుండె-ఊపిరితిత్తుల స్క్రీనింగ్ కేంద్రం"),
                                         ("mr", "हृदय-फुप्फुस तपासणी केंद्र"),
                                         ("ta", "இதயம்-நுரையீரல் பரிசோதனை மையம்")])
def test_every_chiron_language_loads(lang, title):
    at = app()
    at.selectbox(key="m_sex_en").set_value("female").run()
    at.selectbox(key="lang").set_value(lang).run()
    assert not at.exception and len(at.error) == 0
    text = page_text(at)
    assert title in text and "System architecture" not in text
    assert at.selectbox(key=f"m_sex_{lang}").value == "female"


def test_pregnancy_is_greyed_out_for_a_male_patient():
    at = app()
    at.selectbox(key="m_sex_en").set_value("female").run()
    at.selectbox(key="m_preg_en").set_value(True).run()
    assert at.session_state["meta_preg"] is True
    at.selectbox(key="m_sex_en").set_value("male").run()
    assert not at.exception and len(at.error) == 0
    field = at.selectbox(key="m_preg_na_en")
    assert field.disabled and field.value == "not_applicable"
    assert not [s for s in at.selectbox if s.key == "m_preg_en"]
    assert at.session_state["meta_preg"] is None                 # the earlier answer is cleared
    assert "1 detail(s) recorded" in page_text(at)                # sex only: pregnancy is implied, not entered
    at.selectbox(key="m_sex_en").set_value("female").run()
    assert not at.selectbox(key="m_preg_en").disabled and at.selectbox(key="m_preg_en").value is None


def test_reference_cases_start_with_diseased_ones():
    index = json.loads((ROOT / "Test_Data" / "index.json").read_text(encoding="utf-8"))
    for branch, key in (("heart", "cardio"), ("xray", "pulmo")):
        truth = {item["file"]: item["ground_truth"] for item in index[key]["items"]}
        cases = DEMO[branch]
        assert len({c["file"] for c in cases}) == len(cases) > 0
        assert all(truth[c["file"]] == c["ground_truth"] for c in cases)
        assert all((ROOT / "Test_Data" / c["file"]).exists() for c in cases)
        assert all(c["ground_truth"] != "Normal" for c in cases[:5]), branch
        assert any(c["ground_truth"] == "Normal" for c in cases), branch
        assert len(cases) + len(DEMO["left_out"][branch]) == len(truth)
    assert [c["ground_truth"] for c in DEMO["heart"][:4]] == ["AS", "MS", "MR", "MVP"]


@needs_models
def test_visit_analysis_gives_one_decision():
    at = app()
    at.button(key="btn_visit").click().run()
    assert not at.exception and len(at.error) == 0
    text = page_text(at)
    assert "Refer for clinical review" in text            # the first cases of both cards are diseased
    assert "Analysis complete" in text
    assert 'class="ab done' in text                     # finished blocks are marked done
    results = at.session_state["results"]
    for b in ("heart", "xray"):
        assert results[b]["pred"] == results[b]["expected"] == DEMO[b][0]["ground_truth"]
    assert results["heart"]["meta_sig"] is None         # no patient details: no metadata vote made up
    assert "metadata" not in results["heart"]["votes"]
    assert len(at.session_state["records"]) == 2 and not at.session_state["pending"]


@needs_models
def test_next_case_is_analysed():
    at = app()
    at.button(key="next_heart").click().run()
    assert not at.exception and len(at.error) == 0
    assert at.session_state["idx_heart"] == 1
    assert at.session_state["results"]["heart"]["case"] == 2       # numbered as the card's counter shows it
    assert at.session_state["results"]["heart"]["expected"] == DEMO["heart"][1]["ground_truth"]
    assert at.session_state["results"]["xray"] is None  # only the card that moved runs


@needs_models
def test_screening_batch_answers_every_reference_case_correctly():
    at = app()
    at.button(key="btn_batch").click().run()
    assert not at.exception and len(at.error) == 0
    records = at.session_state["records"]
    n_heart, n_xray = len(DEMO["heart"]), len(DEMO["xray"])
    assert len(records) == n_heart + n_xray and all(r["batch"] for r in records)
    wrong = [(r["branch"], r["source_file"], r["finding"], r["expected"]) for r in records if not r["correct"]]
    assert not wrong, f"rerun app/make_demo_cases.py: {wrong}"
    assert all(r["confidence"] >= 0.60 for r in records)
    assert f"{n_heart} of {n_heart} demo cases analysed" in page_text(at)
