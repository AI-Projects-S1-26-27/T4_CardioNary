"""CardioNary dashboard, in the layout of the Chiron screening station.

    streamlit run app/app.py

Ported from the Chiron gateway (gateway/static): the header with language and
screening batch, the two node cards with case navigation, verdict, probability
bars, media, timing line and reporting threshold, the batch accuracy cards, the
confidence history, the case log with CSV export, and the case-detail sections
with Chiron's clinical reference text.

CardioNary adds the patient details, late fusion, one decision per visit and the
architecture of the team's report, drawn live: each block lights up while its
stage of cardio.predict.analyse_heart or lung.predict.analyse_xray runs, and at
rest the diagram marks where the user is.

The reference cases are public held-out data (Test_Data, fold 0 of both models):
the cases listed in app/demo_cases.json, which app/make_demo_cases.py chooses as
the ones the dashboard answers correctly, diseased cases first. Nothing is written
to disk; the case log lives in the browser session.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for _p in (str(REPO_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import streamlit as st

import dashboard_ui as ui
from dashboard_text import LANGS, class_name, clinical, tr
from core.contracts import SYMPTOMS, PatientMetadata, VisitDecision
from cardio.predict import analyse_heart
from lung.predict import analyse_xray

TEST_DATA = REPO_ROOT / "Test_Data"
DEMO_CASES = APP_DIR / "demo_cases.json"
REPORTS = REPO_ROOT / "reports"
MODELS = REPO_ROOT / "models"
HOLD = float(os.environ.get("CARDIONARY_STAGE_HOLD", "0.22"))   # seconds each stage stays lit
BRANCHES = ("heart", "xray")
MANIFEST_KEY = {"heart": "cardio", "xray": "pulmo"}
MAX_RECORDS = 2000
DEFAULT_FLOOR = 0.60          # reporting threshold: below it a result is not conclusive
DEFAULT_COST_RATIO = 4.0      # C_miss / C_refer, so the visit threshold is 1 / (1 + 4) = 0.2
FOCUS = {   # where the user is -> blocks to mark, caption
    "start": (("heart_audio", "metadata", "xray"), "cap_start"),
    "heart": (("heart_audio",), "cap_heart"),
    "xray": (("xray",), "cap_xray"),
    "meta": (("metadata", "meta_classifier"), "cap_meta"),
    "decision": (("decision",), "cap_decision"),
    "done": (("decision", "report"), "cap_done"),
    "heart_done": (("heart_fusion",), "cap_branch_done"),
    "xray_done": (("xray_fusion",), "cap_branch_done"),
    "batch_done": ((), "cap_batch_done"),
}


# ---------------------------------------------------------------------------
# Data that does not change while the app runs
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def manifest() -> dict:
    """The reference cases per branch, in the order the cards show them.

    app/demo_cases.json lists the held-out cases the dashboard answers correctly,
    diseased cases first; without it, every case of Test_Data/index.json is used.
    """
    if DEMO_CASES.exists():
        data = json.loads(DEMO_CASES.read_text(encoding="utf-8"))
        return {b: data.get(b, []) for b in BRANCHES}
    path = TEST_DATA / "index.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {b: data.get(MANIFEST_KEY[b], {}).get("items", []) for b in BRANCHES}


def reference_input(item: dict, idx: int) -> dict:
    """The input of a card for reference case number idx (from 0) of the manifest."""
    return {"kind": "reference", "waiting": False, "idx": idx, "item": item, "path": TEST_DATA / item["file"],
            "name": item.get("source_file") or item["file"], "sig": ("reference", idx)}


@st.cache_data(show_spinner=False)
def reference_info() -> dict:
    """Model sizes and accuracies, read from the committed reports."""
    def load(name):
        try:
            return json.loads((REPORTS / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    qh, cv, qx = (load("quantization_chiron_cardio_v1.json"), load("heart_cv_yaseen.json"),
                  load("quantization_chiron_pulmo_v1.json"))
    return {"bytes": {"heart": qh.get("tflite_bytes"), "xray": qx.get("tflite_bytes")},
            "heart_acc": qh.get("int8_acc"), "heart_cv": cv.get("accuracy_mean"),
            "xray_acc": qx.get("int8_acc"), "xray_pacc": qx.get("int8_patient_acc")}


@st.cache_data(show_spinner=False)
def file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def reference_lines(lang: str) -> dict:
    r = reference_info()
    return {"heart": tr(lang, "ref_heart", acc=ui.pct(r["heart_acc"]), cv=ui.pct(r["heart_cv"])),
            "xray": tr(lang, "ref_xray", acc=ui.pct(r["xray_acc"]), pacc=ui.pct(r["xray_pacc"]))}


def branch_label(lang: str, b: str) -> str:
    return tr(lang, "cardiac" if b == "heart" else "pulmonary")


# ---------------------------------------------------------------------------
# What is on the page now
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    for key, value in (("idx_heart", 0), ("idx_xray", 0), ("results", {"heart": None, "xray": None}),
                       ("pending", []), ("focus", "start"), ("records", []), ("seq", 0)):
        if key not in ss:
            ss[key] = value


def current_input(b: str) -> dict:
    """The input of one card: a reference case, an upload, or nothing yet."""
    ss = st.session_state
    items = manifest()[b]
    if ss.get(f"srcv_{b}", "reference") == "upload":
        up = ss.get(f"up_{b}")
        if up is None:
            return {"kind": "upload", "waiting": True, "sig": ("upload", None)}
        fid = getattr(up, "file_id", None) or f"{up.name}:{up.size}"
        return {"kind": "upload", "waiting": False, "file": up, "name": up.name, "sig": ("upload", fid)}
    if not items:
        return {"kind": "reference", "waiting": True, "sig": ("reference", None)}
    idx = ss.get(f"idx_{b}", 0) % len(items)
    return reference_input(items[idx], idx)


def current_metadata() -> PatientMetadata | None:
    """The patient details as entered; None when nothing was recorded, so no vote is made up.

    A male patient is not pregnant: that is a fact, not a guess, and it is also how
    the training data records every male patient.
    """
    ss = st.session_state
    symptoms = frozenset(s for s in (ss.get("meta_symp") or []) if s in SYMPTOMS)
    sex, pregnant = ss.get("meta_sex"), ss.get("meta_preg")
    sex = sex if sex in ("female", "male") else None
    values = dict(age_years=ss.get("m_age"), sex=sex,
                  height_cm=ss.get("m_height"), weight_kg=ss.get("m_weight"),
                  pregnant=False if sex == "male" else pregnant if isinstance(pregnant, bool) else None)
    if all(v is None for v in values.values()) and not symptoms:
        return None
    return PatientMetadata(**values, symptoms=symptoms)


def details_entered(meta: PatientMetadata | None) -> int:
    """How many patient details were entered; a male patient's pregnancy is implied, not entered."""
    if meta is None:
        return 0
    fields = [meta.age_years, meta.sex, meta.height_cm, meta.weight_kg]
    if meta.sex != "male":
        fields.append(meta.pregnant)
    return sum(v is not None for v in fields) + len(meta.symptoms)


def meta_signature(meta: PatientMetadata | None):
    if meta is None:
        return None
    return (meta.age_years, meta.sex, meta.height_cm, meta.weight_kg, meta.pregnant, tuple(sorted(meta.symptoms)))


def current_result(b: str, inp: dict) -> dict | None:
    r = st.session_state.results.get(b)
    return r if r is not None and r.get("input_sig") == inp["sig"] else None


def visit_summary(results_now: dict, meta_sig) -> dict:
    """One decision per visit from the included cards: p = 1 - prod(1 - p_branch)."""
    ss = st.session_state
    status, p, stale = {}, {}, False
    for b in BRANCHES:
        r = results_now[b]
        if not ss.get(f"inc_{b}", True):
            status[b] = "excluded"
            continue
        if r is None:
            status[b] = "missing"
            continue
        status[b] = "error" if r.get("error") else "withheld" if r.get("withheld") else "ok"
        if status[b] == "ok":
            p[b] = float(r["p"])
        stale = stale or r.get("meta_sig") != meta_sig
    included = [b for b in BRANCHES if status[b] != "excluded"]
    if not included:
        state = "no_tests"
    elif any(status[b] == "missing" for b in included):
        state = "pending"
    elif not p:
        state = "none"
    else:
        state = "made"
    p_any = None
    if state == "made":
        p_any = 1.0 - float(np.prod([1.0 - v for v in p.values()]))
    return {"status": status, "p": p, "p_any": p_any, "state": state, "stale": stale}


def arch_view(inputs: dict, results_now: dict, meta, visit: dict) -> dict:
    ss = st.session_state
    view = {b: {"included": ss.get(f"inc_{b}", True), "waiting": inputs[b]["waiting"], "result": results_now[b]}
            for b in BRANCHES}
    view.update(metadata=meta is not None, decision={"made": "made", "none": "none"}.get(visit["state"]))
    return view


def focus_and_caption(lang: str, inputs: dict):
    f = st.session_state.focus
    blocks, key = FOCUS.get(f, FOCUS["start"])
    if f in BRANCHES and inputs[f]["waiting"]:
        text = tr(lang, "cap_wait_upload", branch=branch_label(lang, f))
    elif f in ("heart_done", "xray_done"):
        text = tr(lang, key, branch=branch_label(lang, f.split("_")[0]))
    else:
        text = tr(lang, key)
    return blocks, ui.esc(text)


# ---------------------------------------------------------------------------
# Callbacks: they only record what to do; the work runs at the end of the script
# ---------------------------------------------------------------------------
def queue(action: str, focus: str | None = None) -> None:
    ss = st.session_state
    if action not in ss.pending:
        ss.pending.append(action)
    if focus:
        ss.focus = focus


def set_focus(focus: str) -> None:
    st.session_state.focus = focus


def step(b: str, delta: int) -> None:
    """Chiron's case navigation: stepping to a case analyses it."""
    n = len(manifest()[b]) or 1
    st.session_state[f"idx_{b}"] = (st.session_state.get(f"idx_{b}", 0) + delta) % n
    queue(b, focus=b)


def remember(canon_key: str, widget_key: str, focus: str) -> None:
    """Keep a choice in a language-independent key.

    Streamlit 1.65 stores selectbox, radio and multiselect values as their displayed
    labels, which change with the language. These widgets therefore carry the
    language in their key, and the choice itself lives in canon_key, which seeds
    the widget again after a language switch.
    """
    st.session_state[canon_key] = st.session_state.get(widget_key)
    st.session_state.focus = focus


def on_sex(widget_key: str) -> None:
    """Choosing male clears any pregnancy answer: the field is then not applicable."""
    remember("meta_sex", widget_key, "meta")
    if st.session_state.get("meta_sex") == "male":
        st.session_state["meta_preg"] = None


def on_source(b: str, widget_key: str) -> None:
    ss = st.session_state
    remember(f"srcv_{b}", widget_key, b)
    if ss.get(f"srcv_{b}") == "reference" or ss.get(f"up_{b}") is not None:
        queue(b)


def on_upload(b: str) -> None:
    st.session_state.focus = b
    if st.session_state.get(f"up_{b}") is not None:
        queue(b)


def clear_records() -> None:
    st.session_state.records = []


# ---------------------------------------------------------------------------
# Running the branches, with the architecture lit stage by stage
# ---------------------------------------------------------------------------
def running_caption(lang: str, stage_name: str) -> str:
    return ui.esc(tr(lang, "cap_running", stage="\x00")).replace("\x00", f"<b>{ui.esc(stage_name)}</b>")


class ArchTracker:
    """Keeps the block states during a run and redraws the diagram at each stage."""

    def __init__(self, lang: str, placeholder, states: dict, metadata_present: bool):
        self.lang, self.ph, self.states = lang, placeholder, dict(states)
        self.metadata_present, self.active = metadata_present, []

    def render(self, caption: str, progress: float | None = None) -> None:
        self.ph.markdown(ui.architecture_html(self.lang, self.states, (), caption, progress), unsafe_allow_html=True)

    def light(self, blocks, caption: str, hold: float = HOLD, progress: float | None = None) -> None:
        for b in self.active:
            if self.states.get(b) == "active":
                self.states[b] = "done"
        if not self.metadata_present:
            blocks = [b for b in blocks if b != "meta_classifier"]
        self.active = list(blocks)
        for b in self.active:
            self.states[b] = "active"
        self.render(caption, progress)
        if hold > 0:
            time.sleep(hold)

    def settle(self, states: dict) -> None:
        self.states.update(states)
        self.active = []

    def reset_branch(self, b: str) -> None:
        for blk in ui.BRANCH_BLOCKS[b]:
            self.states[blk] = "idle"
        self.states[ui.INPUT_BLOCK[b]] = "ready"


class StageClock:
    """Time per stage from the on_stage calls, leaving out the time each stage stays lit."""

    def __init__(self):
        self.ms, self.cur, self.t = {}, None, time.perf_counter()

    def enter(self, stage: str) -> None:
        now = time.perf_counter()
        if self.cur is not None:
            self.ms[self.cur] = self.ms.get(self.cur, 0.0) + (now - self.t) * 1000.0
        self.cur = stage

    def resume(self) -> None:
        self.t = time.perf_counter()

    def stop(self) -> None:
        self.enter(None)


def stage_callback(tracker: ArchTracker, lang: str, b: str, hold: float):
    def on_stage(stage: str) -> None:
        tracker.light(ui.STAGE_BLOCKS[b].get(stage, ()), running_caption(lang, tr(lang, f"stg_{b}_{stage}")), hold)
    return on_stage


def _plain(d: dict | None) -> dict:
    """numpy numbers to Python numbers, so results serialise and compare cleanly."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


def _base(inp: dict) -> dict:
    """What every result records about its input; "case" is the number the card's counter shows."""
    ref = inp["kind"] == "reference"
    return {"input_sig": inp["sig"], "case": inp["idx"] + 1 if ref else None,
            "source_file": inp.get("name"), "expected": inp["item"].get("ground_truth") if ref else None}


def run_heart(inp: dict, meta, on_stage=None, light: bool = False) -> dict:
    out = _base(inp)
    src = str(inp["path"]) if inp["kind"] == "reference" else inp["file"].getvalue()
    res = analyse_heart(src, meta, on_stage=on_stage)
    usable = res.withheld is None and res.model_output is not None
    probs = dict(res.model_output.probabilities) if res.model_output is not None else None
    t = res.timings_ms
    out.update(
        withheld=res.withheld, quality=_plain(res.quality), seconds=res.seconds_of_audio, windows=res.windows,
        temperature=res.temperature, probs=probs if usable else None,
        pred=res.predicted_class if usable else None,
        confidence=(probs[res.predicted_class] if usable else None),
        close_call=res.close_call if usable else None,
        p=res.p_h if usable else None,
        votes=dict(res.fused.votes) if usable else {}, weights=dict(res.fused.weights) if usable else {},
        fusion=res.fusion, characteristics=_plain(res.characteristics), phase=res.systole_or_diastole,
        front_ms=sum(t.get(k, 0.0) for k in ("load_and_resample", "quality_check", "front_end")) if t else None,
        infer_ms=t.get("cnn_8bit"), spectrogram=None)
    if not light and res.audio_8k is not None:
        from core.spectrogram import spectrogram_png
        out["spectrogram"] = spectrogram_png(res.pcm16())
    return out


def run_xray(inp: dict, meta, on_stage=None, light: bool = False) -> dict:
    out = _base(inp)
    src = str(inp["path"]) if inp["kind"] == "reference" else inp["file"].getvalue()
    clock = StageClock()

    def staged(stage: str) -> None:
        clock.enter(stage)
        if on_stage is not None:
            on_stage(stage)
        clock.resume()

    model_out, fused, clahe, occ = analyse_xray(src, meta, on_stage=staged)
    clock.stop()
    probs = dict(model_out.probabilities)
    pred = max(probs, key=probs.get)
    out.update(withheld=None, probs=probs, pred=pred, confidence=probs[pred], close_call=None,
               p=float(fused.p_abnormal), votes=dict(fused.votes), weights=dict(fused.weights),
               secondary=_plain(model_out.secondary), front_ms=clock.ms.get("load_image"),
               infer_ms=clock.ms.get("cnn_8bit"), occlusion=np.asarray(occ, dtype=float).round(4).tolist())
    if not light:
        from core.descriptors import vision_descriptors
        from lung.predict import _load_img_from_source

        img01 = _load_img_from_source(src)
        d = vision_descriptors(img01)
        out.update(img_uri=ui.png_uri(img01), clahe_uri=ui.png_uri(clahe),
                   descriptors={k: d.get(k) for k in ("dense_area_pct", "distribution", "predominant_zone",
                                                      "pattern", "texture_heterogeneity")})
    return out


def run_branch(b: str, inp: dict, meta, on_stage=None, light: bool = False) -> dict:
    try:
        return (run_heart if b == "heart" else run_xray)(inp, meta, on_stage=on_stage, light=light)
    except Exception as exc:          # shown in the card as a warning, never as a red error
        out = _base(inp)
        out.update(error=f"{type(exc).__name__}: {exc}", pred=None, confidence=None, p=None)
        return out


def add_record(b: str, out: dict, batch: bool = False) -> None:
    """One row of the case log, newest first."""
    ss = st.session_state
    ss.seq += 1
    correct = (out["pred"] == out["expected"]) if out.get("expected") and out.get("pred") else None
    ss.records.insert(0, {
        "seq": ss.seq, "time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "branch": b,
        "case": out.get("case"), "source_file": out.get("source_file"), "finding": out.get("pred"),
        "confidence": out.get("confidence"), "expected": out.get("expected"), "correct": correct,
        "p_abnormal": out.get("p"), "front_ms": out.get("front_ms"), "infer_ms": out.get("infer_ms"),
        "withheld": bool(out.get("withheld") or out.get("error")), "batch": batch})
    del ss.records[MAX_RECORDS:]


def branch_final_states(b: str, out: dict, metadata_present: bool) -> dict:
    view = {x: {"included": True, "waiting": False, "result": out if x == b else None} for x in BRANCHES}
    view.update(metadata=metadata_present, decision=None)
    s = ui.resting_states(view)
    return {k: s[k] for k in (*ui.BRANCH_BLOCKS[b], "meta_classifier")}


def run_batch(lang: str, tracker: ArchTracker) -> None:
    """Chiron's screening batch: every reference case of both branches, graded against its label."""
    items = manifest()
    total = sum(len(items[b]) for b in BRANCHES) or 1
    done = 0
    for b in BRANCHES:
        tracker.reset_branch(b)
        blocks, n = list(ui.BRANCH_BLOCKS[b]), len(items[b])
        for i, item in enumerate(items[b]):
            if i % 4 == 0 or i == n - 1:
                caption = ui.esc(tr(lang, "cap_batch", branch=branch_label(lang, b), i=i + 1, n=n))
                tracker.light(blocks, caption, hold=0, progress=done / total)
            add_record(b, run_branch(b, reference_input(item, i), None, light=True), batch=True)
            done += 1
        tracker.settle({k: "done" for k in blocks})
    tracker.render(ui.esc(tr(lang, "cap_batch_done")), progress=1.0)


def run_pending(lang: str, arch_ph, inputs: dict, meta, meta_sig, base_states: dict) -> None:
    ss = st.session_state
    pending = list(ss.pending)
    tracker = ArchTracker(lang, arch_ph, base_states, meta is not None)
    if "batch" in pending:
        run_batch(lang, tracker)
    todo = [b for b in BRANCHES if not inputs[b]["waiting"]
            and (b in pending or ("visit" in pending and ss.get(f"inc_{b}", True)))]
    hold = HOLD if "visit" in pending else HOLD * 0.6
    for b in todo:
        tracker.reset_branch(b)
    for b in todo:
        out = run_branch(b, inputs[b], meta, on_stage=stage_callback(tracker, lang, b, hold))
        out["meta_sig"] = meta_sig
        ss.results[b] = out
        add_record(b, out)
        tracker.settle(branch_final_states(b, out, meta is not None))
    if todo:
        results_now = {b: current_result(b, inputs[b]) for b in BRANCHES}
        visit = visit_summary(results_now, meta_sig)
        if visit["state"] in ("made", "none"):
            tracker.light(("decision",), running_caption(lang, tr(lang, "stg_decision")), hold)
            tracker.light(("report",), running_caption(lang, tr(lang, "stg_report")), hold)
        ss.focus = "done" if "visit" in pending else f"{todo[0]}_done" if len(todo) == 1 else "done"
    elif "batch" in pending:
        ss.focus = "batch_done"
    ss.pending = []
    st.rerun()


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def audio_bytes(inp: dict) -> bytes | None:
    if inp.get("waiting"):
        return None
    return file_bytes(str(inp["path"])) if inp["kind"] == "reference" else inp["file"].getvalue()


def withheld_reason(lang: str, res: dict) -> str:
    if res.get("withheld") == "quality_check":
        q = res.get("quality") or {}
        return tr(lang, "reason_quality_check", hi=q.get("hi_band"), thr=q.get("threshold"))
    return tr(lang, f"reason_{res.get('withheld')}")


def region_text(lang: str, occ) -> str | None:
    if occ is None:
        return None
    a = np.asarray(occ, dtype=float).reshape(3, 3)
    if a.max() <= 0:
        return tr(lang, "none_detected")
    r, c = divmod(int(a.argmax()), 3)
    return tr(lang, "region", zone=tr(lang, ("z_upper", "z_middle", "z_lower")[r]), side=tr(lang, f"side_{c}"))


def show_case(lang: str, b: str, inp: dict, res: dict) -> None:
    """Chiron's case-detail overlay, as a Streamlit dialog."""
    @st.dialog(tr(lang, "case_title"), width="large")
    def dialog():
        if res.get("case") is not None:
            sub = (f'{branch_label(lang, b)} · {tr(lang, "case_n")} {res["case"]} · '   # numbered from 1, as in Chiron
                   f'{tr(lang, "source")}: {res.get("source_file")}')
        else:
            sub = f'{branch_label(lang, b)} · {tr(lang, "upload_tag")}'
        st.markdown(f'<p class="cn-case-sub">{ui.esc(sub)}</p>', unsafe_allow_html=True)
        if b == "heart":
            data = audio_bytes(inp)
            if data:
                st.audio(data, format="audio/wav")
        else:
            st.markdown(ui.xray_media_html(lang, res), unsafe_allow_html=True)
        detail = dict(res)
        if b == "xray":
            detail["region"] = region_text(lang, res.get("occlusion"))
        clin = clinical(lang, b, res["pred"]) if res.get("pred") else None
        st.markdown(ui.case_html(lang, b, detail, clin, reference_lines(lang)[b], tr(lang, f"dataset_{b}")),
                    unsafe_allow_html=True)

    dialog()


def xray_uri(inp: dict) -> str | None:
    """The chest X-ray of a card as a data URI, shown before it is analysed."""
    if inp.get("waiting"):
        return None
    if inp["kind"] == "reference":
        data, mime = file_bytes(str(inp["path"])), mimetypes.guess_type(str(inp["path"]))[0]
    else:
        data, mime = inp["file"].getvalue(), getattr(inp["file"], "type", None)
    return f"data:{mime or 'image/png'};base64," + base64.b64encode(data).decode("ascii")


def render_preview(lang: str, b: str, inp: dict) -> None:
    """The input of a card that is not analysed yet: the heart sound to play, or the X-ray to see."""
    if b == "heart":
        data = audio_bytes(inp)
        if data:
            st.audio(data, format="audio/wav")
            st.markdown(f'<span class="cn-cap">{ui.esc(tr(lang, "media_audio"))}</span>', unsafe_allow_html=True)
    else:
        uri = xray_uri(inp)
        if uri:
            st.markdown(ui.xray_preview_html(lang, uri), unsafe_allow_html=True)
    hint = tr(lang, "preview_hint", run=tr(lang, "run_this"), visit=tr(lang, "analyse_visit"))
    st.markdown(f'<div class="cn-note">{ui.esc(hint)}</div>', unsafe_allow_html=True)


def render_result(lang: str, b: str, inp: dict, res: dict) -> None:
    floor = st.session_state.get(f"floor_{b}", DEFAULT_FLOOR)
    if res.get("withheld"):
        st.markdown(f'<div class="cn-verdict uncertain">{ui.esc(tr(lang, "not_conclusive"))}</div>', unsafe_allow_html=True)
        st.warning(tr(lang, "withheld_warn", reason=withheld_reason(lang, res)))
    else:
        st.markdown(ui.one_line(ui.verdict_html(lang, b, res, floor), ui.bars_html(lang, b, res.get("probs") or {}),
                                ui.fusion_html(lang, b, res)), unsafe_allow_html=True)
    if inp["kind"] == "upload":
        st.markdown(ui.upload_tag_html(lang), unsafe_allow_html=True)
    if b == "heart":
        data = audio_bytes(inp)
        if data:
            st.audio(data, format="audio/wav")
            st.markdown(f'<span class="cn-cap">{ui.esc(tr(lang, "media_audio"))}</span>', unsafe_allow_html=True)
        st.markdown(ui.spectrogram_html(lang, res.get("spectrogram")), unsafe_allow_html=True)
    else:
        st.markdown(ui.xray_media_html(lang, res), unsafe_allow_html=True)


def render_card(lang: str, b: str, inp: dict, res: dict | None) -> None:
    ss = st.session_state
    included = ss.get(f"inc_{b}", True)
    items = manifest()[b]
    if not included:
        st.markdown(ui.dim_card_css(b), unsafe_allow_html=True)
    with st.container(key=f"node_{b}"):
        st.markdown(ui.node_head_html(lang, b), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.25], vertical_alignment="center")
        c1.toggle(tr(lang, "include"), value=True, key=f"inc_{b}", on_change=set_focus, args=("decision",))
        sources = ["reference", "upload"]
        c2.radio(tr(lang, "src_reference"), sources, index=sources.index(inp["kind"]), key=f"src_{b}_{lang}",
                 horizontal=True, label_visibility="collapsed", on_change=on_source, args=(b, f"src_{b}_{lang}"),
                 format_func=lambda v: tr(lang, "src_reference" if v == "reference" else "src_upload"))
        if not included:
            st.markdown(f'<div class="cn-note">{ui.esc(tr(lang, "excluded_note"))}</div>', unsafe_allow_html=True)
        if inp["kind"] == "upload":
            st.file_uploader(tr(lang, "upload_audio" if b == "heart" else "live_image"),
                             type=["wav"] if b == "heart" else ["png", "jpg", "jpeg"], key=f"up_{b}",
                             on_change=on_upload, args=(b,), help=tr(lang, f"upload_hint_{b}"))
        running = b in ss.pending or ("visit" in ss.pending and included and not inp["waiting"])
        if running:
            st.markdown(ui.analysing_html(lang), unsafe_allow_html=True)
        elif inp["waiting"]:
            st.markdown(ui.no_result_html(lang, "upload_waiting"), unsafe_allow_html=True)
        elif res is None:
            st.markdown(ui.no_result_html(lang), unsafe_allow_html=True)
            render_preview(lang, b, inp)
        elif res.get("error"):
            st.warning(tr(lang, "analysis_failed", branch=branch_label(lang, b), err=res["error"]))
        else:
            render_result(lang, b, inp, res)
        st.markdown(ui.meta_line_html(lang, b, None if running else res, reference_info()["bytes"][b]),
                    unsafe_allow_html=True)
        st.slider(tr(lang, "floor"), min_value=0.05, max_value=0.95, value=DEFAULT_FLOOR, step=0.01, key=f"floor_{b}",
                  help=tr(lang, "floor_note"), on_change=set_focus, args=(b,))
        if inp["kind"] == "reference" and items:
            n1, n2, n3 = st.columns([1, 1.1, 1], vertical_alignment="center")
            n1.button(tr(lang, "prev"), key=f"prev_{b}", on_click=step, args=(b, -1), width="stretch")
            n2.markdown(ui.counter_html(lang, inp["idx"], len(items)), unsafe_allow_html=True)
            n3.button(tr(lang, "next"), key=f"next_{b}", on_click=step, args=(b, 1), width="stretch")
        a1, a2 = st.columns(2)
        a1.button(tr(lang, "run_this"), key=f"run_{b}", on_click=queue, args=(b, b), width="stretch",
                  disabled=inp["waiting"])
        if a2.button(tr(lang, "view_case"), key=f"detail_{b}", type="primary", width="stretch",
                     disabled=res is None or bool(res.get("error")) or running):
            show_case(lang, b, inp, res)


def render_patient(lang: str, meta) -> None:
    ss = st.session_state
    with st.container(key="cn_patient"):
        st.markdown(ui.panel_head(tr(lang, "patient_title"), tr(lang, "patient_note")), unsafe_allow_html=True)
        c = st.columns([1, 1.1, 1, 1, 1.1, 2.5])
        common = dict(on_change=set_focus, args=("meta",))
        none_label = tr(lang, "not_recorded")
        c[0].number_input(tr(lang, "age"), min_value=0.0, max_value=120.0, value=None, step=1.0, key="m_age",
                          placeholder=none_label, **common)
        sexes, answers = [None, "female", "male"], [None, True, False]
        sex = ss.get("meta_sex") if ss.get("meta_sex") in sexes else None
        preg = ss.get("meta_preg") if ss.get("meta_preg") in answers else None
        c[1].selectbox(tr(lang, "sex"), sexes, index=sexes.index(sex), key=f"m_sex_{lang}",
                       on_change=on_sex, args=(f"m_sex_{lang}",),
                       format_func=lambda v: none_label if v is None else tr(lang, f"sex_{v}"))
        c[2].number_input(tr(lang, "height"), min_value=20.0, max_value=250.0, value=None, step=1.0, key="m_height",
                          placeholder=none_label, **common)
        c[3].number_input(tr(lang, "weight_kg"), min_value=0.5, max_value=300.0, value=None, step=0.5, key="m_weight",
                          placeholder=none_label, **common)
        if sex == "male":      # greyed out: a male patient cannot be pregnant
            c[4].selectbox(tr(lang, "pregnant"), ["not_applicable"], key=f"m_preg_na_{lang}", disabled=True,
                           format_func=lambda v: tr(lang, v))
        else:
            c[4].selectbox(tr(lang, "pregnant"), answers, index=answers.index(preg), key=f"m_preg_{lang}",
                           on_change=remember, args=("meta_preg", f"m_preg_{lang}", "meta"),
                           format_func=lambda v: none_label if v is None else tr(lang, "yes" if v else "no"))
        c[5].multiselect(tr(lang, "symptoms"), sorted(SYMPTOMS),
                         default=[s for s in (ss.get("meta_symp") or []) if s in SYMPTOMS],
                         key=f"m_symp_{lang}", placeholder=none_label,
                         on_change=remember, args=("meta_symp", f"m_symp_{lang}", "meta"),
                         format_func=lambda s: tr(lang, f"symp_{s}"))
        n = details_entered(meta)
        note = tr(lang, "meta_none") if not n else tr(lang, "meta_some", n=n)
        st.markdown(f'<div class="cn-note">{ui.esc(note)}</div>', unsafe_allow_html=True)


def visit_json(visit: dict, ratio: float, meta, results_now: dict) -> str:
    tau = 1.0 / (1.0 + ratio)
    decision = VisitDecision(refer=visit["p_any"] >= tau, p_any=visit["p_any"], cost_ratio=ratio,
                             p_heart=visit["p"].get("heart"), p_xray=visit["p"].get("xray"))

    def branch(r):
        if r is None:
            return None
        keys = ("case", "source_file", "expected", "withheld", "pred", "confidence", "p", "votes", "weights")
        return {k: r.get(k) for k in keys}

    return json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "CardioNary public demo: a demonstration, not a diagnosis.",
        "decision": {"refer": decision.refer, "p_any": decision.p_any, "threshold": tau,
                     "cost_ratio": decision.cost_ratio, "p_heart": decision.p_heart, "p_xray": decision.p_xray},
        "heart": branch(results_now["heart"]), "xray": branch(results_now["xray"]),
        "patient_details": None if meta is None else {
            "age_years": meta.age_years, "sex": meta.sex, "height_cm": meta.height_cm, "weight_kg": meta.weight_kg,
            "pregnant": meta.pregnant, "symptoms": sorted(meta.symptoms)},
    }, indent=2)


def render_decision(lang: str, visit: dict, meta, results_now: dict) -> None:
    ss = st.session_state
    with st.container(key="cn_decision"):
        st.markdown(ui.panel_head(tr(lang, "decision_title"), tr(lang, "decision_hint")), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2.6], gap="large")
        with c1:
            st.button(tr(lang, "analyse_visit"), type="primary", key="btn_visit", on_click=queue,
                      args=("visit", "done"), width="stretch")
            st.slider(tr(lang, "cost_ratio"), min_value=1.0, max_value=10.0, value=DEFAULT_COST_RATIO, step=0.5,
                      key="cost_ratio",
                      help=tr(lang, "cost_ratio_help"), on_change=set_focus, args=("decision",))
            ratio = float(ss.cost_ratio)
            st.markdown(f'<div class="cn-note">{ui.esc(tr(lang, "threshold_line", tau=f"{1 / (1 + ratio):.3f}"))}</div>',
                        unsafe_allow_html=True)
        with c2:
            if "visit" in ss.pending:
                st.markdown(ui.analysing_html(lang), unsafe_allow_html=True)
                return
            st.markdown(ui.decision_html(lang, visit, ratio), unsafe_allow_html=True)
            if visit["stale"] and visit["state"] in ("made", "none"):
                st.markdown(f'<div class="cn-stale">{ui.esc(tr(lang, "stale_note"))}</div>', unsafe_allow_html=True)
            if visit["state"] == "made":
                st.download_button(tr(lang, "export_visit"), data=visit_json(visit, ratio, meta, results_now),
                                   file_name="cardionary_visit_summary.json", mime="application/json", key="dl_visit")


def batch_stats(records: list) -> dict:
    """Chiron's accuracy cards, over the latest result of each reference case."""
    ss = st.session_state
    out = {}
    for b in BRANCHES:
        latest = {}
        for r in records:
            if r["branch"] == b and r.get("expected") and r.get("case") is not None and r["case"] not in latest:
                latest[r["case"]] = r
        rows = list(latest.values())
        floor = ss.get(f"floor_{b}", DEFAULT_FLOOR)
        conclusive = [r for r in rows if r.get("confidence") is not None and r["confidence"] >= floor]
        ok = sum(1 for r in conclusive if r.get("correct"))
        times = [r["infer_ms"] for r in rows if r.get("infer_ms") is not None]
        out[b] = {"n": len(rows), "total": len(manifest()[b]), "graded": len(conclusive), "ok": ok,
                  "uncertain": len(rows) - len(conclusive),
                  "accuracy": ok / len(conclusive) if conclusive else None,
                  "mean_ms": float(np.mean(times)) if times else 0.0}
    return out


def records_csv(records: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seq", "time_utc", "type", "case", "source_file", "finding", "confidence", "expected", "match",
                "p_abnormal", "signal_ms", "analysis_ms", "withheld", "batch"])
    for r in reversed(records):
        w.writerow([r["seq"], r["time"], r["branch"], r.get("case"), r.get("source_file"), r.get("finding"),
                    None if r.get("confidence") is None else round(r["confidence"], 4), r.get("expected"),
                    r.get("correct"), None if r.get("p_abnormal") is None else round(r["p_abnormal"], 4),
                    None if r.get("front_ms") is None else round(r["front_ms"], 2),
                    None if r.get("infer_ms") is None else round(r["infer_ms"], 2), r.get("withheld"), r.get("batch")])
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="CardioNary", page_icon=ui.favicon(), layout="wide",
                       initial_sidebar_state="collapsed")
    init_state()
    ss = st.session_state
    lang = ss.get("lang", "en")
    st.markdown(ui.CSS, unsafe_allow_html=True)

    inputs = {b: current_input(b) for b in BRANCHES}
    meta = current_metadata()
    meta_sig = meta_signature(meta)
    results_now = {b: current_result(b, inputs[b]) for b in BRANCHES}
    visit = visit_summary(results_now, meta_sig)
    base_states = ui.resting_states(arch_view(inputs, results_now, meta, visit))

    with st.container(key="cn_header"):
        h1, h2, h3 = st.columns([5, 1.4, 1.7], vertical_alignment="center")
        models = {"cardiac": (MODELS / "chiron_cardio_v1_int8.tflite").exists(),
                  "pulmonary": (MODELS / "chiron_pulmo_v1_int8.tflite").exists()}
        h1.markdown(ui.header_html(lang, models), unsafe_allow_html=True)
        h2.selectbox(tr(lang, "language"), list(LANGS), format_func=LANGS.get, key="lang")
        h3.button(tr(lang, "run_selftest"), type="primary", key="btn_batch", on_click=queue, args=("batch",),
                  width="stretch")
    st.markdown(ui.demo_bar_html(lang), unsafe_allow_html=True)

    with st.container(key="cn_arch"):
        arch_ph = st.empty()
    focus, caption = focus_and_caption(lang, inputs)
    if ss.pending:
        focus, caption = (), ui.esc(tr(lang, "analysing"))
    arch_ph.markdown(ui.architecture_html(lang, base_states, focus, caption), unsafe_allow_html=True)

    for col, b in zip(st.columns(2, gap="medium"), BRANCHES):
        with col:
            render_card(lang, b, inputs[b], results_now[b])

    render_patient(lang, meta)
    render_decision(lang, visit, meta, results_now)

    with st.container(key="cn_acc"):
        st.markdown(ui.panel_head(tr(lang, "accuracy_title"), tr(lang, "accuracy_note"))
                    + ui.accuracy_html(lang, batch_stats(ss.records), reference_lines(lang)), unsafe_allow_html=True)
    with st.container(key="cn_chart"):
        st.markdown(ui.panel_head(tr(lang, "history_title"), tr(lang, "history_note")), unsafe_allow_html=True)
        st.iframe(ui.chart_html(lang, ss.records), height=214)
    with st.container(key="cn_log"):
        l1, l2, l3 = st.columns([3.4, 1, 1], vertical_alignment="bottom")
        l1.markdown(ui.panel_head(tr(lang, "log_title")), unsafe_allow_html=True)
        l2.download_button(tr(lang, "export"), data=records_csv(ss.records), file_name="cardionary_case_log.csv",
                           mime="text/csv", key="dl_csv", width="stretch", disabled=not ss.records)
        l3.button(tr(lang, "purge"), key="btn_purge", on_click=clear_records, width="stretch", disabled=not ss.records)
        st.markdown(ui.log_html(lang, ss.records), unsafe_allow_html=True)
    st.markdown(ui.footer_html(lang), unsafe_allow_html=True)

    if ss.pending:
        run_pending(lang, arch_ph, inputs, meta, meta_sig, base_states)


if __name__ == "__main__":
    main()
