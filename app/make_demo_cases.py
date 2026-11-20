"""Choose the reference cases the dashboard shows, and their order.

    python app/make_demo_cases.py

The dashboard is shown in the team's presentation, so its reference cases are the
held-out cases of Test_Data (fold 0 of both models) that it answers correctly. Each
case runs through app.run_branch, the code of the screening batch, with no patient
details, and is kept when:

- nothing is withheld or fails, and the predicted class is the expected one, at or
  above the default reporting threshold (app.DEFAULT_FLOOR);
- the visit decision is right at the default cost ratio (threshold tau): a diseased
  case alone is referred (p >= tau), and a normal heart sound with a normal X-ray is
  not (1 - (1 - p_h)(1 - p_x) < tau). Normal heart sounds are few, so the heart keeps
  every correct one below tau, and the X-ray keeps the normal cases that stay routine
  next to the least certain of them.

Order: five diseased cases first (for the heart, one of each valve disease, then the
next), then normal and diseased cases take turns. The most confident case of a class
comes first.

Writes app/demo_cases.json, with every case left out and why. Run it again when a
model or the fusion parameters change: tests/test_app.py checks that every case it
lists is still answered correctly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CARDIONARY_STAGE_HOLD", "0")
APP_DIR = Path(__file__).resolve().parent
for _p in (str(APP_DIR.parent), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import app  # noqa: E402  (the page itself runs only under streamlit)
from core.contracts import CLASSES  # noqa: E402

FIRST_DISEASED = 5
ITEM_KEYS = ("file", "vector_id", "ground_truth", "source_file")


def class_problem(item: dict, out: dict) -> str | None:
    """Why a case is not answered correctly on its own, or None."""
    if out.get("error"):
        return f"analysis failed: {out['error']}"
    if out.get("withheld"):
        return f"withheld: {out['withheld']}"
    if out["pred"] != item["ground_truth"]:
        return f"predicted {out['pred']} ({out['confidence']:.1%})"
    if out["confidence"] < app.DEFAULT_FLOOR:
        return f"not conclusive: confidence {out['confidence']:.1%}"
    return None


def demo_order(branch: str, cases: list) -> list:
    """Five diseased cases first, then normal and diseased in turn; the most confident first in each class."""
    queues: dict[str, list] = {}
    for item, out in sorted(cases, key=lambda c: -c[1]["confidence"]):
        queues.setdefault(item["ground_truth"], []).append(item)
    diseases = [c for c in CLASSES[branch] if c != "Normal" and queues.get(c)]
    turn = 0

    def next_diseased():
        nonlocal turn
        for _ in range(len(diseases)):
            cls = diseases[turn % len(diseases)]
            turn += 1
            if queues[cls]:
                return queues[cls].pop(0)
        return None

    order = [x for x in (next_diseased() for _ in range(FIRST_DISEASED)) if x is not None]
    normals = queues.get("Normal", [])
    while normals or any(queues[c] for c in diseases):
        if normals:
            order.append(normals.pop(0))
        diseased = next_diseased()
        if diseased is not None:
            order.append(diseased)
    return order


def main() -> int:
    index = json.loads((app.TEST_DATA / "index.json").read_text(encoding="utf-8"))
    tau = 1.0 / (1.0 + app.DEFAULT_COST_RATIO)
    results = {}
    for b in app.BRANCHES:
        items = index[app.MANIFEST_KEY[b]]["items"]
        results[b] = [(item, app.run_branch(b, app.reference_input(item, i), None, light=True))
                      for i, item in enumerate(items)]

    left_out = {b: [] for b in app.BRANCHES}
    correct = {}
    for b in app.BRANCHES:
        correct[b] = []
        for item, out in results[b]:
            problem = class_problem(item, out)
            if problem is None and item["ground_truth"] != "Normal" and out["p"] < tau:
                problem = f"diseased, but p = {out['p']:.3f} is below the visit threshold {tau:.2f}"
            if problem is None and item["ground_truth"] == "Normal" and out["p"] >= tau:
                problem = f"normal, but p = {out['p']:.3f} alone reaches the visit threshold {tau:.2f}"
            if problem:
                left_out[b].append({"file": item["file"], "ground_truth": item["ground_truth"], "reason": problem})
            else:
                correct[b].append((item, out))

    # A normal heart sound with a normal X-ray must stay routine.
    p_h_max = max((out["p"] for item, out in correct["heart"] if item["ground_truth"] == "Normal"), default=0.0)
    x_limit = 1.0 - (1.0 - tau) / (1.0 - p_h_max)
    kept_xray = []
    for item, out in correct["xray"]:
        if item["ground_truth"] == "Normal" and out["p"] >= x_limit:
            left_out["xray"].append({"file": item["file"], "ground_truth": item["ground_truth"],
                                     "reason": f"normal, but p = {out['p']:.3f}: with a normal heart sound "
                                               f"(p up to {p_h_max:.3f}) the visit would be referred"})
        else:
            kept_xray.append((item, out))
    correct["xray"] = kept_xray

    data = {
        "_about": ("Reference cases the dashboard shows, in this order: held-out cases of Test_Data that it "
                   "answers correctly. Chosen by app/make_demo_cases.py; see its docstring for the rule."),
        "rule": {"reporting_threshold": app.DEFAULT_FLOOR, "cost_ratio": app.DEFAULT_COST_RATIO,
                 "visit_threshold": round(tau, 4), "normal_heart_p_max": round(p_h_max, 4),
                 "normal_xray_p_limit": round(x_limit, 4), "first_diseased": FIRST_DISEASED},
    }
    for b in app.BRANCHES:
        data[b] = [{k: item.get(k) for k in ITEM_KEYS} for item in demo_order(b, correct[b])]
    data["left_out"] = left_out
    app.DEMO_CASES.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for b in app.BRANCHES:
        print(f"{b}: {len(data[b])} of {len(results[b])} reference cases kept; first five: "
              + ", ".join(x["ground_truth"] for x in data[b][:5]))
        for x in left_out[b]:
            print(f"   left out {x['file']}: {x['reason']}")
    print(f"wrote {app.DEMO_CASES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
