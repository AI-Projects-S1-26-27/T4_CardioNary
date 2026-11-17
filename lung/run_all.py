"""Regenerate every X-ray result file, in order (roadmap S5, Harsh).

    python -m lung.run_all --list      # the plan, and what this laptop would skip
    python -m lung.run_all             # every step whose code and data are here
    python -m lung.run_all --only votes calibration
    python -m lung.run_all --from fusion_rsna
    python -m lung.run_all --final     # Mon 16 Nov: a step whose code is missing is an error

Each step runs one of the team's scripts as `python -m <module>` in its own process, from the
repository root, in the order the results depend on each other. The run stops at the first
failure. A step is skipped (and listed at the end) when its code is not in this branch, or its
inputs are not on this laptop: the RSNA images and the Kermany cache sit on different laptops, so
no single laptop regenerates everything; files a laptop skipped come from the laptop that has them.

The base CNN is NOT retrained here: its 5-fold run is cached in reports/xray_oof_kermany.csv and
the final model is in models/. Retrain only on purpose, with `python -m lung.model.cv_oof` and
`python -m core.quantize --dataset kermany --arch mobilenet --holdout-fold 0 --augment`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("models/chiron_pulmo_v1_float.keras", "models/chiron_pulmo_v1_int8.tflite",
               "reports/quantization_chiron_pulmo_v1.json")
KERMANY = ("data/cache/kermany_logmel.npz",)
RSNA = ("data/manifests/rsna_manifest.csv", "data/cache/rsna_96")


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()      # files or folders that must be on this laptop
    model: bool = False              # needs the final model files

    @property
    def code(self) -> Path:
        return REPO_ROOT / (self.module.replace(".", "/") + ".py")


STEPS = (
    Step("features_kermany", "lung.model.features", (), ("reports/xray_features_kermany.csv",), KERMANY),
    Step("votes", "lung.fusion.votes", (),
         ("reports/xray_plr_kermany.csv", "reports/xray_plr_kermany.json"),
         ("reports/xray_features_kermany.csv",)),
    Step("calibration", "lung.fusion.calibrate", (),
         ("reports/xray_calibration_kermany.json",),
         ("reports/xray_oof_kermany.csv", "reports/xray_plr_kermany.csv")),
    Step("metadata_vote", "lung.fusion.meta_vote", (),
         ("reports/xray_pm_rsna.csv", "reports/xray_pm_rsna.json"), ("data/manifests/rsna_manifest.csv",)),
    Step("fusion_kermany", "lung.fusion.late_fusion", (),
         ("reports/xray_fused_kermany.csv", "reports/xray_fused_kermany.json"),
         ("reports/xray_oof_kermany.csv", "reports/xray_plr_kermany.csv")),
    Step("rsna_cnn", "lung.eval.run_rsna", (),
         ("reports/xray_cnn_rsna.csv", "reports/xray_rsna.json"), KERMANY + RSNA, model=True),
    Step("rsna_cache", "lung.data.pack_rsna_cache", (), ("data/cache/rsna_96.npz",), RSNA),
    Step("features_rsna", "lung.model.features",
         ("--cache", "data/cache/rsna_96.npz", "--manifest", "data/manifests/rsna_manifest.csv",
          "--output", "reports/xray_features_rsna.csv"),
         ("reports/xray_features_rsna.csv",), ("data/cache/rsna_96.npz",)),
    Step("rsna_adapted", "lung.eval.run_adapted_rsna", (),
         ("reports/xray_cnn_adapted_rsna.csv",), RSNA, model=True),
    Step("fusion_rsna", "lung.eval.run_fusion_rsna", (),
         ("reports/xray_fusion.json",),
         ("reports/xray_cnn_rsna.csv", "reports/xray_features_rsna.csv", "reports/xray_oof_kermany.csv",
          "reports/xray_plr_kermany.csv", "reports/xray_features_kermany.csv")),
    Step("fusion_params", "lung.fusion.late_fusion", ("--params",), ("reports/xray_fusion_params.json",),
         ("reports/xray_oof_kermany.csv", "reports/xray_features_kermany.csv", "reports/xray_plr_kermany.csv")),
    Step("adaptation_synthetic", "lung.tta.bn_adapt",
         ("--out", "reports/xray_tta_synthetic_shift.json"),
         ("reports/xray_tta_synthetic_shift.json",), KERMANY, model=True),
    Step("adaptation", "lung.eval.run_tta", (), ("reports/xray_tta.json",), KERMANY + RSNA, model=True),
    Step("robustness", "lung.eval.run_robustness", (), ("reports/xray_robustness.json",), KERMANY, model=True),
    Step("timing", "lung.eval.timing", ("--out", "reports/xray_timing.json"),
         ("reports/xray_timing.json",), KERMANY, model=True),
    Step("occlusion_parity", "lung.eval.s2_occlusion_and_parity", (),
         ("reports/xray_s2_int8_parity_and_occlusion.json", "reports/xray_occlusion_8bit.csv"),
         KERMANY + RSNA, model=True),
    Step("edge_budget", "lung.eval.edge_budget", (), ("reports/xray_edge_budget.json",), KERMANY, model=True),
)
NOT_YET = "code not in this branch yet"


def skip_reason(step: Step, produced=frozenset()) -> str | None:
    """None when the step can run on this laptop, otherwise why it is skipped.
    A file that an earlier step of the same run writes counts as present."""
    if not step.code.exists():
        return NOT_YET
    if step.model and not all((REPO_ROOT / f).exists() for f in MODEL_FILES):
        return "model files missing: " + ", ".join(f for f in MODEL_FILES if not (REPO_ROOT / f).exists())
    missing = [n for n in step.needs if n not in produced and not (REPO_ROOT / n).exists()]
    if missing:
        return "data not on this laptop: " + ", ".join(missing)
    return None


def plan(only=None, start: str | None = None) -> list[tuple[Step, str | None]]:
    """The steps to consider, in order, each with None (it runs) or the reason it is skipped."""
    names = [s.name for s in STEPS]
    for name in list(only or []) + ([start] if start else []):
        if name not in names:
            raise ValueError(f"unknown step {name!r}; the steps are: {', '.join(names)}")
    steps = list(STEPS[names.index(start):] if start else STEPS)
    if only:
        steps = [s for s in steps if s.name in only]
    out, produced = [], set()
    for s in steps:
        why = skip_reason(s, produced)
        if why is None:
            produced.update(s.writes)
        out.append((s, why))
    return out


def command(step: Step) -> list[str]:
    return [sys.executable, "-m", step.module, *step.args]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate every X-ray result file, in order.")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--from", dest="start", metavar="STEP", help="start at this step, e.g. after a failure")
    ap.add_argument("--final", action="store_true", help="treat a step whose code is missing as an error")
    args = ap.parse_args(argv)
    try:
        steps = plan(args.only, args.start)
    except ValueError as exc:
        ap.error(str(exc))

    width = max(len(s.name) for s, _ in steps)
    if args.list:
        for s, why in steps:
            print(f"{s.name:<{width}}  {'runs' if why is None else 'skipped: ' + why}")
            print(f"{'':<{width}}  python -m {s.module} {' '.join(s.args)}  ->  {', '.join(s.writes)}")
        return 0

    missing_code = [s.name for s, why in steps if why == NOT_YET]
    if args.final and missing_code:
        print("--final: these steps have no code in this branch: " + ", ".join(missing_code))
        return 2

    done = []
    for s, why in steps:
        if why is not None:
            continue
        cmd = command(s)
        print(f"\n=== {s.name}: python -m {' '.join(cmd[2:])}", flush=True)
        t0 = time.perf_counter()
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            print(f"\n{s.name} failed; stopping. Fix it, then resume with: python -m lung.run_all --from {s.name}")
            return 1
        done.append((s.name, time.perf_counter() - t0))

    print("\n=== Summary")
    for name, seconds in done:
        print(f"{name:<{width}}  ran in {seconds:.0f} s")
    for s, why in steps:
        if why is not None:
            print(f"{s.name:<{width}}  skipped: {why}")
    regenerated = {w for s, why in steps if why is None for w in s.writes}
    left = sorted({w for s, why in steps if why is not None for w in s.writes} - regenerated)
    if left:
        print("Not regenerated on this laptop: " + ", ".join(left))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
