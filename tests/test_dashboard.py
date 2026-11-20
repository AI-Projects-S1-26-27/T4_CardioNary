"""The dashboard's architecture panel and the stage hooks behind it.

- cardio.predict.analyse_heart and lung.predict.analyse_xray report each stage, in
  order, through on_stage, without changing their results;
- every stage maps to blocks of the architecture;
- the block states at rest follow the page (excluded, waiting, withheld, done);
- the text is complete in all six languages, with no emoji and no em dash.
"""

from __future__ import annotations

import io
import re
import string
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import dashboard_text as text  # noqa: E402
import dashboard_ui as ui  # noqa: E402
from cardio import predict  # noqa: E402

HEART_MODEL = ROOT / "models" / "chiron_cardio_v1_int8.tflite"
XRAY_MODEL = ROOT / "models" / "chiron_pulmo_v1_int8.tflite"
XRAY_CASE = ROOT / "Test_Data" / "pulmo_images" / "vec119_pneumonia.png"


def heartbeat_wav(seconds=6.0, fs=4000) -> bytes:
    x = np.zeros(int(seconds * fs))
    for k in range(int(seconds / 0.83) + 1):
        for start, dur, freq, amp in ((0.1, 0.10, 50.0, 0.6), (0.4, 0.08, 70.0, 0.4)):
            n0, n = int((k * 0.83 + start) * fs), int(dur * fs)
            n = min(n, len(x) - n0)
            if n > 0:
                x[n0:n0 + n] += amp * np.hanning(n) * np.sin(2 * np.pi * freq * np.arange(n) / fs)
    buf = io.BytesIO()
    wavfile.write(buf, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Stage hooks
# ---------------------------------------------------------------------------
def test_heart_stages_stop_where_the_recording_is_withheld():
    seen = []
    assert predict.analyse_heart(b"not a wav file", on_stage=seen.append).withheld == "unreadable"
    assert seen == ["load_and_resample"]
    seen.clear()
    assert predict.analyse_heart(heartbeat_wav(1.0), on_stage=seen.append).withheld == "too_short"
    assert seen == ["load_and_resample", "quality_check"]


@pytest.mark.skipif(not HEART_MODEL.exists(), reason="model files not in this checkout")
def test_heart_stages_run_in_order_and_change_nothing():
    seen = []
    data = heartbeat_wav()
    with_hook = predict.analyse_heart(data, on_stage=seen.append)
    assert seen == list(predict.STAGES)
    assert with_hook.model_output.probabilities == predict.analyse_heart(data).model_output.probabilities


@pytest.mark.skipif(not (XRAY_MODEL.exists() and XRAY_CASE.exists()), reason="X-ray model or case not here")
def test_xray_stages_run_in_order_and_change_nothing():
    from lung.predict import XRAY_STAGES, analyse_xray

    seen = []
    m1, f1, _, o1 = analyse_xray(str(XRAY_CASE), on_stage=seen.append)
    m2, f2, _, o2 = analyse_xray(str(XRAY_CASE))
    assert seen == list(XRAY_STAGES)
    assert m1.probabilities == m2.probabilities and f1.p_abnormal == f2.p_abnormal
    assert np.allclose(o1, o2)


def test_every_stage_lights_known_blocks():
    from lung.predict import XRAY_STAGES

    assert set(ui.STAGE_BLOCKS["heart"]) == set(predict.STAGES)
    assert set(ui.STAGE_BLOCKS["xray"]) == set(XRAY_STAGES)
    for stages in ui.STAGE_BLOCKS.values():
        for blocks in stages.values():
            assert blocks and set(blocks) <= set(ui.ALL_BLOCKS)


# ---------------------------------------------------------------------------
# Block states at rest
# ---------------------------------------------------------------------------
def view(heart=None, xray=None, metadata=False, decision=None, **flags):
    base = lambda b: {"included": flags.get(f"{b}_included", True), "waiting": flags.get(f"{b}_waiting", False)}
    return {"heart": {**base("heart"), "result": heart}, "xray": {**base("xray"), "result": xray},
            "metadata": metadata, "decision": decision}


def test_resting_states_follow_the_page():
    s = ui.resting_states(view())
    assert s["heart_audio"] == s["xray"] == "ready" and s["metadata"] == "waiting"
    assert s["meta_classifier"] == "unused" and s["decision"] == "idle"

    done = {"votes": {"cnn": 0.2, "features": 0.1, "metadata": 0.3}}
    s = ui.resting_states(view(heart=done, metadata=True, decision="made", xray_included=False))
    assert all(s[b] == "done" for b in ui.HEART_BLOCKS)
    assert all(s[b] == "unused" for b in ui.XRAY_BLOCKS)
    assert s["meta_classifier"] == "done" and s["decision"] == s["report"] == "done"

    s = ui.resting_states(view(heart={"withheld": "quality_check"}, decision="none", xray_waiting=True))
    assert s["heart_audio"] == "withheld" and s["heart_cnn"] == "unused"
    assert s["xray"] == "waiting" and s["decision"] == "withheld"

    s = ui.resting_states(view(xray={"votes": {"cnn": 0.9}}))
    assert s["xray_cnn"] == "done" and s["xray_features"] == "unused"


def test_edges_flow_into_the_running_block():
    assert ui.edge_state("done", "active") == "active"
    assert ui.edge_state("done", "done") == "done"
    assert ui.edge_state("unused", "done") == "unused"
    assert ui.edge_state("idle", "active") == "idle"


def test_architecture_svg_marks_states_and_focus():
    states = {b: "idle" for b in ui.ALL_BLOCKS}
    states["heart_cnn"] = "active"
    html = ui.architecture_html("en", states, focus=("decision",), caption="x")
    assert html.count("<svg") == 1 and "\n" not in html
    assert 'class="ab active cnn"' in html and 'class="ab idle focus"' in html
    assert ui.architecture_html("hi", states).count("<svg") == 1


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
EMOJI = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + chr(0x2B00) + "-" + chr(0x2BFF) + chr(0xFE0F) + "]")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)


def test_text_is_complete_without_emoji_or_em_dash():
    assert list(text.LANGS) == ["en", "hi", "bn", "te", "mr", "ta"]        # the Chiron station's six
    for lang in text.LANGS:
        assert set(text.APP[lang]) == set(text.APP["en"]), lang
        assert set(text.chiron()[lang]) == {"ui", "clinical"}, lang
    for lang in text.LANGS:
        for value in text.APP[lang].values():
            for s in strings(value):
                assert not EMOJI.search(s) and chr(0x2014) not in s, s
    for name in ("app.py", "dashboard_ui.py", "dashboard_text.py"):
        source = (ROOT / "app" / name).read_text(encoding="utf-8")
        assert not EMOJI.search(source), name
    for lang in text.LANGS:                       # Chiron strings the case details show
        for key in ("generic_note", "phenotype_note", "vision_nondiscrim", "no_measurements"):
            assert chr(0x2014) not in text.tr(lang, key), (lang, key)


def test_translations_keep_placeholders_and_block_shapes():
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}
    for lang in text.LANGS:
        for key, en in text.APP["en"].items():
            value = text.APP[lang][key]
            if isinstance(en, tuple):          # architecture blocks: (bold lines, normal lines)
                assert isinstance(value, tuple) and len(value) == 2 and all(value), (lang, key)
            else:
                assert fields(value) == fields(en), (lang, key)
        assert text.class_name(lang, "heart", "MS") != "MS", lang      # clinical names exist in every language
        assert text.tr(lang, "media_audio") != "Source heart sound" or lang == "en", lang


def test_clinical_text_follows_the_class_order():
    assert text.class_name("en", "heart", "AS") == "Aortic stenosis"
    assert text.class_name("en", "xray", "Pneumonia") == "Pattern consistent with pneumonia"
    assert text.class_name("hi", "heart", "Normal") != "Normal"
    assert "3 seconds" not in text.clinical("en", "heart", "Normal")["limits"]
