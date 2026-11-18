"""Look and markup of the dashboard, ported from the Chiron station (gateway/static).

style.css, the node cards, the accuracy cards, the confidence chart, the case log
and the case-detail sections follow gateway/static/{style.css, app.js}. Two
changes: red is gone (the Chiron accent was red, and misses were red), replaced by
blue for actions and amber for anything that needs attention; and the
architecture of the team's report is drawn as SVG, with each block coloured by
what it is doing.

Everything here returns strings. app.py decides when to show them.
"""

from __future__ import annotations

import base64
import html
import io
import json

import numpy as np

from dashboard_text import LANGS, class_name, tr

# ---------------------------------------------------------------------------
# Palette: Chiron's, without red
# ---------------------------------------------------------------------------
BG, PANEL, PANEL_2, LINE = "#0d1117", "#161b22", "#1c232c", "#2a3038"
TEXT, MUTED = "#e6edf3", "#8b949e"
ACCENT = "#2f81f7"        # Chiron used Silicon Labs red here
OK, WARN = "#2ea043", "#d29922"
CARDIO, PULMO = "#bc8cff", "#58a6ff"   # Chiron's cardiac colour was salmon
ACTIVE = "#39c5cf"


def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def pct(v, digits: int = 1) -> str:
    return "-" if v is None else f"{100.0 * float(v):.{digits}f}%"


def one_line(*parts: str) -> str:
    """Join markup into one line: Streamlit's Markdown ends a raw HTML block at a blank line."""
    return "".join(p.strip() for p in parts).replace("\n", " ")


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
CSS = """
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel-2:#1c232c;--line:#2a3038;--text:#e6edf3;--muted:#8b949e;
--accent:#2f81f7;--ok:#2ea043;--warn:#d29922;--cardio:#bc8cff;--pulmo:#58a6ff;--active:#39c5cf;}
.stApp{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1500px;padding-top:1.6rem;padding-bottom:2.5rem;}
.st-key-cn_header{background:linear-gradient(90deg,#12171e 0%,#171d26 100%);border:1px solid var(--line);
 border-bottom:2px solid var(--accent);border-radius:12px;padding:14px 22px 12px;}
.cn-brand .product{color:var(--accent);font-size:30px;font-weight:700;letter-spacing:2.5px;line-height:1.1;}
.cn-brand .product-desc{display:block;font-size:15px;font-weight:500;letter-spacing:1.4px;color:#b9c4d0;margin-top:2px;}
.cn-brand .sub{margin:4px 0 0;color:var(--muted);font-size:13px;}
.cn-pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;}
.cn-pill{display:flex;align-items:center;gap:7px;padding:4px 12px;border-radius:999px;background:var(--panel-2);
 border:1px solid var(--line);font-size:12.5px;color:var(--text);}
.cn-dot{width:9px;height:9px;border-radius:50%;background:var(--muted);display:inline-block;}
.cn-dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok);}
.cn-dot.off{background:var(--warn);}
.cn-demo-bar{padding:8px 16px;background:var(--warn);color:#1b1300;font-weight:600;font-size:13px;border-radius:8px;}
.cn-demo-bar b{margin-right:10px;letter-spacing:.8px;}
.st-key-cn_arch,.st-key-cn_patient,.st-key-cn_decision,.st-key-cn_acc,.st-key-cn_chart,.st-key-cn_log{
 background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;}
.st-key-node_heart,.st-key-node_xray{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;}
.st-key-node_heart{border-top:3px solid var(--cardio);} .st-key-node_xray{border-top:3px solid var(--pulmo);}
.cn-panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.cn-panel-head .t{font-size:15px;text-transform:uppercase;letter-spacing:1.4px;color:#c9d4e0;font-weight:600;}
.cn-panel-head .hint{color:var(--muted);font-size:12px;}
.cn-node-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;}
.cn-node-head .t{font-size:17px;letter-spacing:.5px;font-weight:600;color:var(--text);}
.cn-node-head .board{color:var(--muted);font-size:12px;}
.cn-verdict{margin:4px 0 4px;font-size:30px;font-weight:700;line-height:1.15;color:var(--text);}
.cn-verdict.uncertain{color:var(--warn);}
.cn-verdict.none,.cn-verdict.analysing{color:var(--muted);font-size:20px;font-weight:400;}
.cn-conf{color:var(--muted);font-size:14px;margin-bottom:2px;}
.cn-match{font-weight:700;font-size:14px;}
.cn-match.ok{color:var(--ok);} .cn-match.bad{color:var(--warn);}
.cn-bars{margin:12px 0 8px;}
.cn-bar-row{display:grid;grid-template-columns:minmax(120px,190px) 1fr 58px;gap:9px;align-items:center;margin:5px 0;font-size:13px;color:var(--text);}
.cn-bar-track{background:#0b0f14;border-radius:4px;height:15px;overflow:hidden;display:block;}
.cn-bar-fill{display:block;height:100%;background:#1f6f33;border-radius:4px;transition:width .25s ease;}
.cn-bar-row.top .cn-bar-fill{background:var(--ok);}
.cn-bar-val{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;}
.cn-fused{font-size:13.5px;color:var(--text);margin:8px 0 2px;}
.cn-fused b{font-variant-numeric:tabular-nums;}
table.cn-votes{width:100%;border-collapse:collapse;font-size:12.5px;margin:4px 0 2px;border:none;}
table.cn-votes td{padding:3px 6px;border:none;border-bottom:1px solid #1e242c;font-variant-numeric:tabular-nums;color:var(--text);}
table.cn-votes td.k{color:var(--muted);} table.cn-votes td.n{text-align:right;}
.cn-meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);
 padding-top:10px;margin-top:4px;}
.cn-meta .c{color:#b9c4d0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.cn-scan{height:3px;border-radius:2px;background:#0b0f14;overflow:hidden;margin:10px 0 4px;}
.cn-scan span{display:block;height:100%;width:38%;border-radius:2px;background:var(--accent);animation:cnscan 1.05s ease-in-out infinite;}
@keyframes cnscan{0%{transform:translateX(-100%);}100%{transform:translateX(300%);}}
.cn-counter{display:flex;justify-content:center;align-items:baseline;gap:5px;padding:6px 10px;border-radius:6px;
 border:1px solid var(--line);background:var(--panel-2);}
.cn-counter b{font-size:16px;font-variant-numeric:tabular-nums;color:var(--text);}
.cn-counter small{font-size:11px;color:var(--muted);}
.cn-media{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 4px;align-items:flex-start;}
.cn-media figure{margin:0;display:flex;flex-direction:column;gap:6px;}
.cn-media img.px{width:168px;height:168px;object-fit:contain;image-rendering:pixelated;background:#0b0f14;border:1px solid #223;border-radius:6px;}
.cn-cap{font-size:11px;color:var(--muted);letter-spacing:.02em;max-width:172px;}
.cn-spec{width:100%;height:auto;background:#05080b;border:1px solid #223;border-radius:6px;display:block;}
.cn-occ{display:grid;grid-template-columns:repeat(3,56px);grid-template-rows:repeat(3,56px);border:1px solid #223;border-radius:6px;overflow:hidden;background:#0b0f14;}
.cn-occ div{display:flex;align-items:center;justify-content:center;font-size:11px;color:#dbe7f2;font-variant-numeric:tabular-nums;border:1px solid rgba(255,255,255,.04);}
.cn-tag{display:inline-block;margin-top:6px;padding:2px 8px;border:1px solid var(--warn);border-radius:3px;color:var(--warn);font-size:11px;}
.cn-note{color:var(--muted);font-size:12.5px;}
.cn-stale{color:var(--warn);font-size:13px;margin:6px 0 2px;}
.cn-accuracy{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:4px;}
.cn-acc-card{background:var(--panel-2);border-radius:9px;padding:14px 16px;}
.cn-acc-card .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1px;}
.cn-acc-card .value{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text);}
.cn-acc-card .detail{color:var(--muted);font-size:13px;}
.cn-acc-card .detail.cv{margin-top:4px;color:#7d8894;}
.cn-table-wrap{overflow-x:auto;max-height:340px;overflow-y:auto;}
table.cn-log{width:100%;border-collapse:collapse;font-size:13px;border:none;}
table.cn-log th{position:sticky;top:0;background:var(--panel);text-align:left;padding:7px 9px;color:var(--muted);font-weight:600;
 text-transform:uppercase;font-size:11px;letter-spacing:.8px;border:none;border-bottom:1px solid var(--line);}
table.cn-log td{padding:6px 9px;border:none;border-bottom:1px solid #1e242c;font-variant-numeric:tabular-nums;color:var(--text);}
table.cn-log td.tick{color:var(--ok);font-weight:700;} table.cn-log td.cross{color:var(--warn);font-weight:700;}
.cn-ttag{padding:1px 7px;border-radius:4px;font-size:11px;font-weight:600;}
.cn-ttag.heart{background:#2a1f3d;color:var(--cardio);} .cn-ttag.xray{background:#16283f;color:var(--pulmo);}
.cn-btag{margin-left:6px;color:var(--muted);font-size:10.5px;}
.cn-decision{border-radius:10px;padding:16px 20px;margin:2px 0 4px;border:1px solid var(--line);background:var(--panel-2);}
.cn-decision.refer{background:#2a2110;border-color:var(--warn);}
.cn-decision.refer .h{color:#f2cc60;}
.cn-decision.routine{background:#12261a;border-color:var(--ok);}
.cn-decision.routine .h{color:#56d364;}
.cn-decision .h{font-size:22px;font-weight:700;margin-bottom:4px;color:var(--text);}
.cn-decision .p{font-size:14px;color:var(--text);}
.cn-decision.none .p{color:var(--muted);}
.cn-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:2px 0 12px;}
.cn-kpi{background:var(--panel-2);border-radius:9px;padding:10px 14px;}
.cn-kpi .label{color:var(--muted);font-size:12.5px;letter-spacing:.4px;}
.cn-kpi .value{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text);}
.cn-kpi .value.muted{font-size:15px;font-weight:400;color:var(--muted);padding-top:6px;}
.cn-case-sub{color:var(--muted);font-size:13px;margin:0 0 6px;}
.cn-case-sec{padding:14px 0;border-bottom:1px solid #1e242c;}
.cn-case-sec:last-child{border-bottom:none;}
.cn-case-sec .h{margin:0 0 10px;font-size:12px;letter-spacing:1.2px;text-transform:uppercase;color:#c9d4e0;font-weight:600;}
.cn-case-sec.guidance{border-left:3px solid var(--accent);padding-left:14px;}
.cn-case-sec.limits{border-left:3px solid var(--warn);padding-left:14px;}
.cn-case-sec.phenotype{border-left:3px solid var(--pulmo);padding-left:14px;}
.cn-kv{display:grid;grid-template-columns:210px 1fr;gap:10px;padding:5px 0;align-items:start;font-size:14px;}
.cn-kv .k{color:var(--muted);} .cn-kv .v{color:var(--text);}
.cn-kv .k::first-letter{text-transform:uppercase;}
ol.cn-actions{margin:0;padding-left:20px;} ol.cn-actions li{margin:3px 0;}
.cn-muted{color:var(--muted);} .cn-small{font-size:12px;}
.cn-footer{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;padding:12px 4px;border-top:1px solid var(--line);
 color:var(--muted);font-size:12px;margin-top:6px;}
.cn-footer .spacer{flex:1;}
.arch-cap{font-size:14px;color:var(--text);margin:6px 0 8px;min-height:22px;}
.arch-cap b{color:var(--active);font-weight:600;}
.arch-progress{height:4px;background:#0b0f14;border-radius:2px;overflow:hidden;margin:0 0 8px;}
.arch-progress span{display:block;height:100%;background:var(--active);}
.arch-scroll{overflow-x:auto;}
.arch-svg{display:block;width:100%;min-width:760px;height:auto;}
.arch-legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:8px;font-size:12px;color:var(--muted);}
.arch-legend .sw{display:inline-block;width:16px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px;border:2px solid;}
.arch-svg text{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,"Nirmala UI","Noto Sans Devanagari",sans-serif;}
.ab rect.fr{fill:#1c232c;stroke:#3a434e;stroke-width:1.6;}
.ab text.t{fill:#e6edf3;font-weight:700;font-size:14.5px;}
.ab text.s{fill:#a3adb8;font-size:12.5px;}
.ab text.f{fill:#c9d4e0;font-size:12.5px;}
.ab.cnn rect.fr{fill:#e6edf3;stroke:#e6edf3;}
.ab.cnn text.t,.ab.cnn text.s{fill:#0d1117;}
.ab.waiting rect.fr{stroke:#8b949e;stroke-width:1.8;stroke-dasharray:6 4;}
.ab.ready rect.fr{stroke:#58a6ff;stroke-width:2.4;}
.ab.active rect.fr{fill:#0f2f36;stroke:#39c5cf;stroke-width:3;animation:abPulse 1.1s ease-in-out infinite;}
.ab.cnn.active rect.fr{fill:#39c5cf;stroke:#39c5cf;}
.ab.done rect.fr{fill:#12261a;stroke:#2ea043;stroke-width:2.2;}
.ab.cnn.done rect.fr{fill:#3fb950;stroke:#3fb950;}
.ab.withheld rect.fr{fill:#2a2110;stroke:#d29922;stroke-width:2.4;stroke-dasharray:7 4;}
.ab.unused{opacity:.3;}
.ab rect.ring{display:none;fill:none;stroke:#58a6ff;stroke-width:1.6;stroke-dasharray:4 3;}
.ab.focus rect.ring{display:inline;}
.ab.focus rect.fr{filter:drop-shadow(0 0 7px rgba(88,166,255,.95));}
@keyframes abPulse{0%,100%{filter:drop-shadow(0 0 2px rgba(57,197,207,.55));}50%{filter:drop-shadow(0 0 13px rgba(57,197,207,1));}}
.ae{fill:none;stroke:#4a5561;stroke-width:1.6;}
.ae.active{stroke:#39c5cf;stroke-width:2.4;stroke-dasharray:7 5;animation:aeFlow .7s linear infinite;}
.ae.done{stroke:#2ea043;stroke-width:2;}
.ae.unused{opacity:.3;}
@keyframes aeFlow{to{stroke-dashoffset:-12;}}
.arch-def{fill:#c9d4e0;font-size:13px;}
.arch-lbl{fill:#c9d4e0;font-size:14px;font-style:italic;}
@media (prefers-reduced-motion: reduce){.ab.active rect.fr,.ae.active,.cn-scan span{animation:none;}}
@media (max-width:620px){.cn-kv{grid-template-columns:1fr;gap:2px;}}
</style>
"""


def dim_card_css(branch: str) -> str:
    """Chiron's offline style, for a card that is left out of the visit."""
    return f"<style>.st-key-node_{branch}{{opacity:.55;}}</style>"


# ---------------------------------------------------------------------------
# Architecture: the figure of the team's report, as live SVG
# ---------------------------------------------------------------------------
BOX = {   # block id: x, y, width, height, in the coordinates of the report's figure
    "heart_secondary": (162, 30, 150, 86),
    "heart_features": (336, 40, 164, 76),
    "heart_audio": (35, 146, 104, 82),
    "spectrogram": (162, 146, 150, 82),
    "heart_cnn": (336, 146, 164, 82),
    "heart_fusion": (523, 146, 118, 82),
    "metadata": (35, 260, 277, 80),
    "meta_classifier": (336, 260, 164, 80),
    "xray": (35, 373, 104, 82),
    "grayscale": (162, 373, 150, 82),
    "xray_cnn": (336, 373, 164, 82),
    "xray_fusion": (523, 373, 118, 82),
    "xray_secondary": (162, 485, 150, 86),
    "xray_features": (336, 485, 164, 80),
    "decision": (668, 188, 222, 232),
    "report": (912, 243, 150, 124),
}
CNN_BLOCKS = {"heart_cnn", "xray_cnn"}
ALL_BLOCKS = tuple(BOX)
HEART_BLOCKS = ("heart_audio", "spectrogram", "heart_cnn", "heart_secondary", "heart_features", "heart_fusion")
XRAY_BLOCKS = ("xray", "grayscale", "xray_cnn", "xray_secondary", "xray_features", "xray_fusion")
BRANCH_BLOCKS = {"heart": HEART_BLOCKS, "xray": XRAY_BLOCKS}
INPUT_BLOCK = {"heart": "heart_audio", "xray": "xray"}
FEATURE_BLOCK = {"heart": "heart_features", "xray": "xray_features"}

# Which blocks run in each stage of cardio.predict.analyse_heart and lung.predict.analyse_xray.
STAGE_BLOCKS = {
    "heart": {
        "load_and_resample": ("heart_audio",),
        "quality_check": ("heart_audio",),
        "front_end": ("spectrogram",),
        "cnn_8bit": ("heart_cnn",),
        "secondary_characteristics": ("heart_secondary",),
        "systole_or_diastole": ("heart_secondary",),
        "votes_and_fusion": ("heart_features", "meta_classifier", "heart_fusion"),
    },
    "xray": {
        "load_image": ("xray", "grayscale"),
        "cnn_8bit": ("xray_cnn",),
        "secondary_characteristics": ("xray_secondary",),
        "occlusion_map": ("xray_secondary", "xray_cnn"),   # the focus region probes the same CNN
        "votes_and_fusion": ("xray_features", "meta_classifier", "xray_fusion"),
    },
}

EDGES = (   # from, to, polyline
    ("heart_audio", "spectrogram", ((139, 187), (160, 187))),
    ("spectrogram", "heart_cnn", ((312, 187), (334, 187))),
    ("heart_cnn", "heart_fusion", ((500, 187), (521, 187))),
    ("heart_audio", "heart_secondary", ((87, 146), (87, 73), (160, 73))),
    ("heart_secondary", "heart_features", ((312, 78), (334, 78))),
    ("heart_features", "heart_fusion", ((500, 78), (582, 78), (582, 144))),
    ("metadata", "meta_classifier", ((312, 300), (334, 300))),
    ("meta_classifier", "heart_fusion", ((500, 300), (582, 300), (582, 230))),
    ("meta_classifier", "xray_fusion", ((500, 300), (582, 300), (582, 371))),
    ("xray", "grayscale", ((139, 414), (160, 414))),
    ("grayscale", "xray_cnn", ((312, 414), (334, 414))),
    ("xray_cnn", "xray_fusion", ((500, 414), (521, 414))),
    ("xray", "xray_secondary", ((87, 455), (87, 528), (160, 528))),
    ("xray_secondary", "xray_features", ((312, 528), (334, 528))),
    ("xray_features", "xray_fusion", ((500, 528), (582, 528), (582, 457))),
    ("heart_fusion", "decision", ((641, 200), (666, 232))),
    ("xray_fusion", "decision", ((641, 400), (666, 376))),
    ("decision", "report", ((890, 305), (910, 305))),
)
MARKER_FILL = {"idle": "#4a5561", "active": ACTIVE, "done": OK, "unused": "#4a5561"}
STATE_ORDER = ("waiting", "ready", "active", "done", "withheld", "unused")


def edge_state(src: str, dst: str) -> str:
    """An edge flows into a running block, is done between finished ones, and fades with an unused end."""
    if "unused" in (src, dst):
        return "unused"
    if dst == "active" and src in ("ready", "active", "done"):
        return "active"
    if src == "done" and dst in ("done", "withheld"):
        return "done"
    return "idle"


def resting_states(view: dict) -> dict:
    """Block states between runs, from what the page holds.

    view = {"heart": {...}, "xray": {...}, "metadata": bool, "decision": "made" | "none" | None},
    each branch {"included": bool, "waiting": bool, "result": dict or None (for the current input)}.
    """
    s = {b: "idle" for b in ALL_BLOCKS}
    metadata_vote = False
    for branch in ("heart", "xray"):
        v = view[branch]
        blocks, inp = BRANCH_BLOCKS[branch], INPUT_BLOCK[branch]
        r = v.get("result")
        if not v.get("included", True):
            s.update({b: "unused" for b in blocks})
        elif v.get("waiting"):
            s[inp] = "waiting"
        elif r is None:
            s[inp] = "ready"
        elif r.get("error") or r.get("withheld"):
            s[inp] = "withheld"
            s.update({b: "unused" for b in blocks[1:]})
        else:
            s.update({b: "done" for b in blocks})
            votes = r.get("votes") or {}
            if "features" not in votes:
                s[FEATURE_BLOCK[branch]] = "unused"
            metadata_vote = metadata_vote or "metadata" in votes
    s["metadata"] = "ready" if view.get("metadata") else "waiting"
    s["meta_classifier"] = "done" if metadata_vote else ("idle" if view.get("metadata") else "unused")
    decision = view.get("decision")
    s["decision"] = {"made": "done", "none": "withheld"}.get(decision, "idle")
    s["report"] = "done" if decision in ("made", "none") else "idle"
    return s


def _segments(segs, cls: str, x: float, y: float, anchor: str = "middle") -> str:
    """One SVG text line with subscripts: segs = [(text, is_subscript), ...].

    Subscripts move with dy, which every browser supports (Firefox ignores baseline-shift).
    """
    out, shifted = [], False
    for text, sub in segs:
        if sub and not shifted:
            out.append(f'<tspan dy="3.5" font-size="9.5">{esc(text)}</tspan>')
            shifted = True
        elif sub:
            out.append(f'<tspan font-size="9.5">{esc(text)}</tspan>')
        elif shifted:
            out.append(f'<tspan dy="-3.5">{esc(text)}</tspan>')
            shifted = False
        else:
            out.append(f'<tspan>{esc(text)}</tspan>')
    return f'<text class="{cls}" x="{x}" y="{y:.1f}" text-anchor="{anchor}">{"".join(out)}</text>'


def _block(lang: str, bid: str, state: str, focus) -> str:
    x, y, w, h = BOX[bid]
    bold, normal = tr(lang, f"blk_{bid}")
    lines = [("t", [(t, False)]) for t in bold] + [("s", [(t, False)]) for t in normal]
    if bid == "decision":
        lines += [("gap", []),
                  ("f", [("p = 1 − (1 − p", False), ("h", True), (")(1 − p", False), ("x", True), (")", False)]),
                  ("f", [(tr(lang, "refer_if") + " p · C", False), ("miss", True), (" >", False)]),
                  ("f", [("(1 − p) · C", False), ("refer", True)])]
    lh = 17.0
    top = y + (h - lh * len(lines)) / 2 + 12.5
    cls = f"ab {state}" + (" cnn" if bid in CNN_BLOCKS else "") + (" focus" if bid in focus else "")
    title = " ".join(bold + normal[:1])
    out = [f'<g class="{cls}"><title>{esc(title)}: {esc(tr(lang, "lg_" + state)) if state != "idle" else ""}</title>',
           f'<rect class="ring" x="{x - 6}" y="{y - 6}" width="{w + 12}" height="{h + 12}" rx="13"/>',
           f'<rect class="fr" x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>']
    for i, (kind, segs) in enumerate(lines):
        if kind != "gap":
            out.append(_segments(segs, kind, x + w / 2, top + i * lh))
    out.append("</g>")
    return "".join(out)


def architecture_svg(lang: str, states: dict, focus=()) -> str:
    parts = ['<svg class="arch-svg" viewBox="20 20 1062 560" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="{esc(tr(lang, "arch_title"))}"><defs>']
    for name, fill in MARKER_FILL.items():
        parts.append(f'<marker id="cnar-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
                     f'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
                     f'<path d="M0,0 L10,5 L0,10 z" fill="{fill}"/></marker>')
    parts.append("</defs>")
    for src, dst, pts in EDGES:
        st_ = edge_state(states.get(src, "idle"), states.get(dst, "idle"))
        d = "M" + " L".join(f"{px},{py}" for px, py in pts)
        parts.append(f'<path class="ae {st_}" d="{d}" marker-end="url(#cnar-{st_})"/>')
    parts.append(_segments([("p", False), ("h", True)], "arch-lbl", 652, 176, "start"))
    parts.append(_segments([("p", False), ("x", True)], "arch-lbl", 652, 438, "start"))
    for bid in BOX:
        parts.append(_block(lang, bid, states.get(bid, "idle"), focus))
    defs = ((("p", False), ("h", True)), tr(lang, "def_ph")), ((("p", False), ("x", True)), tr(lang, "def_px")), \
           ((("C", False), ("miss", True)), tr(lang, "def_cmiss")), ((("C", False), ("refer", True)), tr(lang, "def_crefer"))
    for i, (sym, meaning) in enumerate(defs):
        y = 44 + i * 25
        parts.append(_segments(list(sym), "arch-def", 668, y, "start"))
        parts.append(f'<text class="arch-def" x="712" y="{y}">=  {esc(meaning)}</text>')
    parts.append("</svg>")
    return "".join(parts)


LEGEND = (("waiting", "#1c232c", "#8b949e"), ("ready", "#1c232c", "#58a6ff"), ("active", "#0f2f36", ACTIVE),
          ("done", "#12261a", OK), ("withheld", "#2a2110", WARN), ("unused", "#1c232c", "#3a434e"))


def architecture_html(lang: str, states: dict, focus=(), caption: str = "", progress: float | None = None) -> str:
    """The whole architecture panel as one HTML block: heading, live caption, diagram, legend."""
    legend = "".join(f'<span><span class="sw" style="background:{bg};border-color:{fg}"></span>'
                     f'{esc(tr(lang, "lg_" + name))}</span>' for name, bg, fg in LEGEND)
    legend += (f'<span><span class="sw" style="background:transparent;border-color:#58a6ff;border-style:dashed">'
               f'</span>{esc(tr(lang, "lg_focus"))}</span>'
               f'<span><span class="sw" style="background:#e6edf3;border-color:#e6edf3"></span>{esc(tr(lang, "lg_cnn"))}</span>'
               f'<span><span class="sw" style="background:#1c232c;border-color:#3a434e"></span>{esc(tr(lang, "lg_classic"))}</span>')
    bar = ("" if progress is None else
           f'<div class="arch-progress"><span style="width:{max(0.0, min(1.0, progress)) * 100:.1f}%"></span></div>')
    return one_line(
        f'<div class="cn-panel-head"><span class="t">{esc(tr(lang, "arch_title"))}</span>'
        f'<span class="hint">{esc(tr(lang, "arch_hint"))}</span></div>',
        f'<div class="arch-cap">{caption}</div>', bar,
        f'<div class="arch-scroll">{architecture_svg(lang, states, focus)}</div>',
        f'<div class="arch-legend">{legend}</div>')


# ---------------------------------------------------------------------------
# Header, notices, footer
# ---------------------------------------------------------------------------
def header_html(lang: str, models: dict) -> str:
    pills = "".join(
        f'<span class="cn-pill"><span class="cn-dot {"on" if ok else "off"}"></span>'
        f'{esc(tr(lang, "model_ready" if ok else "model_missing", name=tr(lang, label)))}</span>'
        for label, ok in models.items())
    return one_line(
        '<div class="cn-brand"><div class="product">CardioNary'
        f'<span class="product-desc">{esc(tr(lang, "title"))}</span></div>'
        f'<p class="sub">{esc(tr(lang, "subtitle"))}</p>'
        f'<div class="cn-pills">{pills}</div></div>')


def demo_bar_html(lang: str) -> str:
    return f'<div class="cn-demo-bar"><b>{esc(tr(lang, "demo_label"))}</b>{esc(tr(lang, "public_demo_note"))}</div>'


def panel_head(title: str, hint: str = "") -> str:
    return (f'<div class="cn-panel-head"><span class="t">{esc(title)}</span>'
            f'<span class="hint">{esc(hint)}</span></div>')


def footer_html(lang: str) -> str:
    return one_line(f'<div class="cn-footer"><span>{esc(tr(lang, "footer_repo"))}</span><span class="spacer"></span>'
                    f'<span>{esc(tr(lang, "retention"))}</span></div>')


# ---------------------------------------------------------------------------
# Node cards
# ---------------------------------------------------------------------------
def node_head_html(lang: str, branch: str) -> str:
    key, board = ("cardiac_node", "model_heart") if branch == "heart" else ("pulmonary_node", "model_xray")
    return (f'<div class="cn-node-head"><span class="t">{esc(tr(lang, key))}</span>'
            f'<span class="board">{esc(tr(lang, board))}</span></div>')


def analysing_html(lang: str) -> str:
    return (f'<div class="cn-verdict analysing">{esc(tr(lang, "analysing"))}</div>'
            '<div class="cn-scan"><span></span></div>')


def no_result_html(lang: str, text_key: str = "no_result") -> str:
    return f'<div class="cn-verdict none">{esc(tr(lang, text_key))}</div>'


def bars_html(lang: str, modality: str, probs: dict) -> str:
    """Chiron's probability bars: sorted, the winner highlighted, every value printed."""
    if not probs:
        return ""
    top = max(probs, key=probs.get)
    rows = []
    for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        rows.append(f'<div class="cn-bar-row{" top" if cls == top else ""}"><span>{esc(class_name(lang, modality, cls))}</span>'
                    f'<span class="cn-bar-track"><span class="cn-bar-fill" style="width:{max(p * 100, 1):.1f}%"></span></span>'
                    f'<span class="cn-bar-val">{pct(p)}</span></div>')
    return f'<div class="cn-bars">{"".join(rows)}</div>'


def verdict_html(lang: str, modality: str, res: dict, floor: float) -> str:
    """Verdict, confidence, close call and the match against the expected answer."""
    conf = res.get("confidence")
    uncertain = conf is None or conf < floor
    name = tr(lang, "not_conclusive") if uncertain else class_name(lang, modality, res.get("pred"))
    out = [f'<div class="cn-verdict{" uncertain" if uncertain else ""}">{esc(name)}</div>',
           f'<div class="cn-conf">{esc(tr(lang, "confidence"))} {pct(conf)}'
           + (f' · {esc(class_name(lang, modality, res.get("pred")))}' if uncertain and res.get("pred") else "")
           + "</div>"]
    cc = res.get("close_call")
    if cc:
        out.append(f'<div class="cn-conf">{esc(tr(lang, "close_call", a=class_name(lang, modality, cc[0]), b=class_name(lang, modality, cc[1])))}</div>')
    expected = res.get("expected")
    if expected and res.get("pred") and not uncertain:
        exp_name = class_name(lang, modality, expected)
        if res["pred"] == expected:
            out.append(f'<div class="cn-match ok">{esc(tr(lang, "matches"))} · {esc(exp_name)}</div>')
        else:
            out.append(f'<div class="cn-match bad">{esc(tr(lang, "differs"))} · {esc(tr(lang, "expected"))}: {esc(exp_name)}</div>')
    elif expected:
        out.append(f'<div class="cn-conf">{esc(tr(lang, "expected"))}: {esc(class_name(lang, modality, expected))}</div>')
    return "".join(out)


def fusion_html(lang: str, modality: str, res: dict) -> str:
    """The branch probability after late fusion, and each vote with its weight."""
    p, votes, weights = res.get("p"), res.get("votes") or {}, res.get("weights") or {}
    if p is None:
        return ""
    what = tr(lang, "what_heart" if modality == "heart" else "what_xray")
    rows = "".join(
        f'<tr><td class="k">{esc(tr(lang, "vote_" + name))}</td><td class="n">{pct(votes[name])}</td>'
        f'<td class="n">{esc(tr(lang, "weight"))} {weights.get(name, 0.0):.2f}</td></tr>'
        for name in ("cnn", "features", "metadata") if name in votes)
    out = (f'<div class="cn-fused">{esc(tr(lang, "fused_line", p=pct(p), what=what))}</div>'
           f'<table class="cn-votes"><tbody>{rows}</tbody></table>')
    if set(votes) == {"cnn"}:
        out += f'<div class="cn-note">{esc(tr(lang, "fusion_fallback"))}</div>'
    return out


def meta_line_html(lang: str, modality: str, res: dict | None, model_bytes: int | None) -> str:
    front = "-" if not res or res.get("front_ms") is None else f'{res["front_ms"]:.1f} ms'
    infer = "-" if not res or res.get("infer_ms") is None else f'{res["infer_ms"]:.1f} ms'
    size = "-" if not model_bytes else f"{model_bytes:,} B"
    inp = tr(lang, "input_heart" if modality == "heart" else "input_xray")
    return one_line(
        f'<div class="cn-meta"><span>{esc(tr(lang, "front_time"))} <span class="c">{front}</span></span>',
        f'<span>{esc(tr(lang, "infer_time"))} <span class="c">{infer}</span></span>',
        f'<span>{esc(tr(lang, "model_size"))} <span class="c">{size}</span></span>',
        f'<span>{esc(tr(lang, "input"))} <span class="c">{esc(inp)}</span></span></div>')


def counter_html(lang: str, idx: int, total: int) -> str:
    return (f'<div class="cn-counter"><small>{esc(tr(lang, "case_n"))}</small><b>{idx + 1 if total else 0}</b>'
            f'<small>{esc(tr(lang, "of"))} {total}</small></div>')


def spectrogram_html(lang: str, uri: str | None) -> str:
    if not uri:
        return ""
    return (f'<div class="cn-media" style="display:block"><img class="cn-spec" src="{uri}" alt="{esc(tr(lang, "spectrogram"))}">'
            f'<div class="cn-cap" style="max-width:none;margin-top:6px">{esc(tr(lang, "spectrogram"))}</div></div>')


def png_uri(img) -> str:
    """A grayscale image (uint8 0..255, or float 0..1) as a PNG data URI."""
    from PIL import Image

    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(np.round(a.astype(np.float64) * (255.0 if a.max() <= 1.0 else 1.0)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def occlusion_grid_html(occ) -> str:
    """The 3 x 3 occlusion map: blue where hiding that ninth lowers the probability, with the value printed."""
    a = np.asarray(occ, dtype=float).reshape(3, 3)
    top = max(float(a.max()), 1e-6)
    cells = []
    for v in a.ravel():
        alpha = max(0.0, float(v)) / top
        cells.append(f'<div style="background:rgba(88,166,255,{0.08 + 0.72 * alpha:.2f})">{v:+.2f}</div>')
    return f'<div class="cn-occ">{"".join(cells)}</div>'


def xray_media_html(lang: str, res: dict) -> str:
    figs = []
    if res.get("img_uri"):
        figs.append(f'<figure><img class="px" src="{res["img_uri"]}" alt=""><span class="cn-cap">{esc(tr(lang, "media_input_xray"))}</span></figure>')
    if res.get("clahe_uri"):
        figs.append(f'<figure><img class="px" src="{res["clahe_uri"]}" alt=""><span class="cn-cap">{esc(tr(lang, "clahe_label"))}</span></figure>')
    if res.get("occlusion") is not None:
        figs.append(f'<figure>{occlusion_grid_html(res["occlusion"])}<span class="cn-cap">{esc(tr(lang, "occlusion_label"))}</span></figure>')
    return f'<div class="cn-media">{"".join(figs)}</div>' if figs else ""


def upload_tag_html(lang: str) -> str:
    return f'<span class="cn-tag">{esc(tr(lang, "upload_tag"))}</span>'


# ---------------------------------------------------------------------------
# Visit decision
# ---------------------------------------------------------------------------
def decision_html(lang: str, visit: dict, ratio: float) -> str:
    tau = 1.0 / (1.0 + ratio)

    def kpi(label_key, value, muted=False):
        return (f'<div class="cn-kpi"><div class="label">{esc(tr(lang, label_key))}</div>'
                f'<div class="value{" muted" if muted else ""}">{esc(value)}</div></div>')

    def branch_value(b):
        status = visit["status"][b]
        if status == "ok":
            return kpi("lbl_heart" if b == "heart" else "lbl_xray", pct(visit["p"][b]))
        word = {"excluded": "not_in_visit", "withheld": "withheld_short", "error": "withheld_short"}.get(status, "not_analysed")
        return kpi("lbl_heart" if b == "heart" else "lbl_xray", tr(lang, word), muted=True)

    p_any = visit.get("p_any")
    kpis = (branch_value("heart") + branch_value("xray")
            + kpi("lbl_any", pct(p_any) if p_any is not None else "-", muted=p_any is None)
            + kpi("lbl_tau", pct(tau)))
    state = visit["state"]
    if state == "made":
        refer = p_any >= tau
        cls, head = ("refer", "decision_refer") if refer else ("routine", "decision_routine")
        body = tr(lang, head + "_desc", p_any=pct(p_any), tau=pct(tau), ratio=f"{ratio:g}")
        box = f'<div class="cn-decision {cls}"><div class="h">{esc(tr(lang, head))}</div><div class="p">{esc(body)}</div></div>'
    else:
        msg = {"none": "no_usable_test", "no_tests": "no_tests_included"}.get(state, "no_analysis_yet")
        box = f'<div class="cn-decision none"><div class="p">{esc(tr(lang, msg))}</div></div>'
    return one_line(f'<div class="cn-kpis">{kpis}</div>', box)


# ---------------------------------------------------------------------------
# Batch results, confidence history, case log
# ---------------------------------------------------------------------------
def accuracy_html(lang: str, stats: dict, reference: dict) -> str:
    cards = []
    for b, label_key in (("heart", "cardiac"), ("xray", "pulmonary")):
        s = stats[b]
        ref = reference.get(b, "")
        if not s["n"]:
            cards.append(f'<div class="cn-acc-card"><div class="label">{esc(tr(lang, label_key))}</div>'
                         f'<div class="value">-</div><div class="detail">{esc(tr(lang, "no_result"))}</div>'
                         f'<div class="detail cv">{esc(ref)}</div></div>')
            continue
        value = pct(s["accuracy"]) if s["accuracy"] is not None else "-"
        cards.append(
            f'<div class="cn-acc-card"><div class="label">{esc(tr(lang, label_key))}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="detail">{s["ok"]} / {s["graded"]} {esc(tr(lang, "correct"))} · '
            f'{s["uncertain"]} {esc(tr(lang, "not_conclusive"))} · {s["mean_ms"]:.1f} ms {esc(tr(lang, "mean_time"))}</div>'
            f'<div class="detail">{esc(tr(lang, "acc_cases", n=s["n"], total=s["total"]))}</div>'
            f'<div class="detail cv">{esc(ref)}</div></div>')
    return f'<div class="cn-accuracy">{"".join(cards)}</div>'


def chart_html(lang: str, records: list) -> str:
    """Chiron's confidence-history canvas: last 60 analyses per branch, misses marked (amber, not red)."""
    data = {}
    for b in ("heart", "xray"):
        pts = [r for r in records if r["branch"] == b and r.get("confidence") is not None][:60]
        data[b] = [{"c": round(float(r["confidence"]), 4), "ok": None if r.get("correct") is None else int(r["correct"])}
                   for r in reversed(pts)]
    labels = {"heart": tr(lang, "cardiac"), "xray": tr(lang, "pulmonary"), "miss": tr(lang, "legend_miss"),
              "empty": tr(lang, "log_empty")}
    return """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:%(panel)s;color:%(muted)s;font:12px -apple-system,"Segoe UI",Roboto,Helvetica,Arial,"Nirmala UI",sans-serif;color-scheme:dark;}
#chart{display:block;width:100%%;height:180px;}
.lg{display:flex;gap:16px;margin:6px 2px 0;}
.lg i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px;}
.lg b{display:inline-block;width:8px;height:8px;border-radius:50%%;vertical-align:middle;margin-right:6px;background:%(warn)s;}
</style></head><body>
<canvas id="chart" height="180"></canvas>
<div class="lg"><span><i style="background:%(cardio)s"></i>%(l_heart)s</span><span><i style="background:%(pulmo)s"></i>%(l_xray)s</span><span><b></b>%(l_miss)s</span></div>
<script>
const DATA = %(data)s, EMPTY = %(empty)s;
const COL = {heart: '%(cardio)s', xray: '%(pulmo)s'};
function draw() {
  const cv = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(200, Math.min(cv.clientWidth || 600, 4000)), h = 180;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const pad = 26;
  g.strokeStyle = '%(line)s'; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h - 2 * pad) * i / 4;
    g.beginPath(); g.moveTo(pad, y); g.lineTo(w - 8, y); g.stroke();
    g.fillStyle = '%(muted)s'; g.font = '10px sans-serif'; g.fillText((100 - i * 25) + '%%', 2, y + 3);
  }
  let any = false;
  for (const b of ['heart', 'xray']) {
    const pts = DATA[b]; if (!pts.length) continue; any = true;
    const X = i => pad + (w - pad - 10) * (pts.length > 1 ? i / (pts.length - 1) : 0.5);
    const Y = c => pad + (h - 2 * pad) * (1 - c);
    g.strokeStyle = COL[b]; g.lineWidth = 2; g.beginPath();
    pts.forEach((e, i) => i ? g.lineTo(X(i), Y(e.c)) : g.moveTo(X(i), Y(e.c)));
    g.stroke();
    pts.forEach((e, i) => {
      if (pts.length === 1) { g.fillStyle = COL[b]; g.beginPath(); g.arc(X(i), Y(e.c), 3, 0, 7); g.fill(); }
      if (e.ok === 0) { g.fillStyle = '%(warn)s'; g.beginPath(); g.arc(X(i), Y(e.c), 3.5, 0, 7); g.fill(); }
    });
  }
  if (!any) { g.fillStyle = '%(muted)s'; g.font = '13px sans-serif'; g.textAlign = 'center'; g.fillText(EMPTY, w / 2, h / 2 + 4); }
}
window.addEventListener('resize', draw); draw();
</script></body></html>""" % {
        "panel": PANEL, "muted": MUTED, "line": LINE, "warn": WARN, "cardio": CARDIO, "pulmo": PULMO,
        "data": json.dumps(data), "empty": json.dumps(labels["empty"]),
        "l_heart": esc(labels["heart"]), "l_xray": esc(labels["xray"]), "l_miss": esc(labels["miss"]),
    }


def log_html(lang: str, records: list) -> str:
    """Chiron's case log, newest first."""
    if not records:
        return f'<div class="cn-note">{esc(tr(lang, "log_empty"))}</div>'
    head = "".join(f"<th>{esc(tr(lang, k))}</th>" for k in ("th_time", "th_node", "th_predicted", "th_conf", "th_expected"))
    head += "<th></th>" + "".join(f"<th>{esc(tr(lang, k))}</th>" for k in ("th_case", "th_front", "th_infer"))
    rows = []
    for r in records[:200]:
        b = r["branch"]
        tag = f'<span class="cn-ttag {b}">{esc(tr(lang, "cardiac" if b == "heart" else "pulmonary"))}</span>'
        if r.get("batch"):
            tag += f'<span class="cn-btag">{esc(tr(lang, "batch_tag"))}</span>'
        finding = (tr(lang, "withheld_short") if r.get("withheld") else class_name(lang, b, r.get("finding")))
        mark = ("<td></td>" if r.get("correct") is None else
                f'<td class="tick">{esc(tr(lang, "matches"))}</td>' if r["correct"] else
                f'<td class="cross">{esc(tr(lang, "differs"))}</td>')
        case = "-" if r.get("case") is None else str(r["case"])
        front = "-" if r.get("front_ms") is None else f'{r["front_ms"]:.1f} ms'
        infer = "-" if r.get("infer_ms") is None else f'{r["infer_ms"]:.1f} ms'
        expected = class_name(lang, b, r["expected"]) if r.get("expected") else "-"
        rows.append(f'<tr><td>{esc(r["time"])}</td><td>{tag}</td><td>{esc(finding)}</td><td>{pct(r.get("confidence"))}</td>'
                    f'<td>{esc(expected)}</td>{mark}<td>{esc(case)}</td><td>{front}</td><td>{infer}</td></tr>')
    return one_line(f'<div class="cn-table-wrap"><table class="cn-log"><thead><tr>{head}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Case details: Chiron's sections, measured first, reference text marked as such
# ---------------------------------------------------------------------------
def fmt_energy(v) -> str:
    """The share of sound energy above 1 kHz, as a plain decimal (no 1.1e-05)."""
    return "-" if v is None else f"{float(v):.5f}"


def kv(label: str, value) -> str:
    label = label[:1].upper() + label[1:]          # Chiron's "confidence" doubles as an inline word
    return f'<div class="cn-kv"><span class="k">{esc(label)}</span><span class="v">{esc(value)}</span></div>'


def case_html(lang: str, modality: str, res: dict, clin: dict | None, reference_line: str, dataset: str) -> str:
    T = lambda k, **f: tr(lang, k, **f)
    pred, expected = res.get("pred"), res.get("expected")
    model = [kv(T("finding"), class_name(lang, modality, pred) if pred else T("withheld_short")),
             kv(T("confidence"), pct(res.get("confidence")))]
    if expected:
        model.append(kv(T("expected"), class_name(lang, modality, expected)))
    model.append(bars_html(lang, modality, res.get("probs") or {}))
    if res.get("p") is not None:
        model.append(fusion_html(lang, modality, res))

    measured = []
    if modality == "heart":
        ch = res.get("characteristics") or {}
        hr, reg = ch.get("heart_rate_bpm"), ch.get("regularity")
        measured.append(kv(T("hr"), f"{hr:.0f} {T('bpm')}" if hr else T("not_measured")))
        measured.append(kv(T("regularity"), f"{100 * reg:.0f}%" if reg is not None else T("not_measured")))
        q = res.get("quality") or {}
        if q:
            key = "quality_pass" if q.get("in_distribution") else "quality_fail"
            measured.append(kv(T("quality"), T(key, hi=fmt_energy(q.get("hi_band")), thr=fmt_energy(q.get("threshold")))))
        measured.append(kv(T("duration"), f'{res.get("seconds", 0):.1f} {T("seconds")}'))
        if res.get("windows"):
            measured.append(kv(T("windows"), res["windows"]))
        if res.get("temperature"):
            measured.append(kv(T("temperature"), f'{res["temperature"]:.3f}'))
    else:
        measured.append(kv(T("image_size"), "96 × 96"))
        measured.append(f'<p class="cn-muted">{esc(T("no_measurements"))}</p>')

    pheno = []
    if modality == "heart":
        ch = res.get("characteristics") or {}
        if ch.get("segmentation_ok"):
            yn = lambda v: T("not_measured") if v is None else (T("yes") if v else T("no"))
            pheno.append(kv(T("murmur"), yn(ch.get("murmur_detected"))))
            if ch.get("murmur_detected"):
                phase = {"systole": T("systole"), "diastole": T("diastole")}.get(res.get("phase"), T("none_detected"))
                pheno.append(kv(T("murmur_phase"), phase))
                if ch.get("murmur_duration_pct") is not None:
                    pheno.append(kv(T("murmur_dur"), f'{ch["murmur_duration_pct"]:.0f}%'))
                if ch.get("murmur_intensity_db") is not None:
                    pheno.append(kv(T("murmur_int"), f'{ch["murmur_intensity_db"]:.1f} dB'))
            if ch.get("spectral_centroid_hz"):
                pheno.append(kv(T("spectral"), f'{ch["spectral_centroid_hz"]:.0f} Hz'))
            if ch.get("s1_s2_ratio_db") is not None:
                pheno.append(kv(T("s1s2"), f'{ch["s1_s2_ratio_db"]:.1f} dB'))
            if ch.get("systole_ms"):
                pheno.append(kv(T("sys_dur"), f'{ch["systole_ms"]:.0f} ms'))
            if ch.get("diastole_ms"):
                pheno.append(kv(T("dia_dur"), f'{ch["diastole_ms"]:.0f} ms'))
            if ch.get("click_suspected"):
                pheno.append(kv(T("click"), T("suspected")))
    else:
        d = res.get("descriptors") or {}
        if d:
            if res.get("region"):
                pheno.append(kv(T("attention"), res["region"]))
            pheno.append(kv(T("dense_area"), f'{d.get("dense_area_pct")}%'))
            side = {"bilateral": "d_bilateral", "patient right": "d_patient_right", "patient left": "d_patient_left"}
            pheno.append(kv(T("distribution_lbl"), T(side.get(d.get("distribution"), "not_measured"))))
            pheno.append(kv(T("zone"), T("z_" + d.get("predominant_zone", "middle"))))
            pheno.append(kv(T("pattern"), T("p_" + d.get("pattern", "none"))))
            pheno.append(kv(T("texture"), d.get("texture_heterogeneity")))
            pheno.append(f'<p class="cn-muted cn-small">{esc(T("vision_nondiscrim"))}</p>')

    guidance = ""
    if clin:
        actions = "".join(f"<li>{esc(a)}</li>" for a in clin.get("actions", []))
        guidance = (kv(T("finding"), clin.get("finding")) + kv(T("urgency"), clin.get("urgency"))
                    + f'<div class="cn-kv"><span class="k">{esc(T("actions"))}</span><span class="v"><ol class="cn-actions">{actions}</ol></span></div>'
                    + kv(T("confirm"), clin.get("confirm")))
        assoc = (clin.get("associations") or "").strip()
        if assoc and assoc not in ("-", chr(0x2014)):          # Chiron writes a dash for "none"
            guidance += kv(T("associations"), assoc)

    sections = [f'<div class="cn-case-sec"><div class="h">{esc(T("sec_model"))}</div>{"".join(model)}</div>',
                f'<div class="cn-case-sec"><div class="h">{esc(T("sec_measured"))}</div>{"".join(measured)}</div>']
    if pheno:
        sections.append(f'<div class="cn-case-sec phenotype"><div class="h">{esc(T("sec_phenotype"))}</div>'
                        f'<p class="cn-muted cn-small">{esc(T("phenotype_note"))}</p>{"".join(pheno)}</div>')
    sections.append(f'<div class="cn-case-sec guidance"><div class="h">{esc(T("sec_guidance"))}</div>'
                    f'<p class="cn-muted cn-small">{esc(T("generic_note"))}</p>{guidance}</div>')
    sections.append(f'<div class="cn-case-sec limits"><div class="h">{esc(T("sec_limits"))}</div>'
                    f'<p>{esc(clin.get("limits", "") if clin else "")}</p>'
                    f'{kv(T("cv_note"), reference_line)}{kv(T("dataset"), dataset)}</div>')
    return one_line(*sections)


# ---------------------------------------------------------------------------
# Favicon: a pulse line, not an emoji
# ---------------------------------------------------------------------------
def favicon():
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((2, 2, 62, 62), radius=14, fill=PANEL, outline=ACCENT, width=3)
    d.line([(9, 34), (21, 34), (26, 22), (32, 46), (38, 16), (43, 34), (55, 34)], fill=PULMO, width=5, joint="curve")
    return im


__all__ = ["LANGS"]
