"""S4 (Harsh): the adapted CNN's probabilities on RSNA, for the generalisation comparison.

Re-estimates the batch-normalisation statistics on N unlabelled RSNA images (default 32,
random draw, seed 0; lung.tta.bn_adapt.apply_bn_adaptation, which never changes the original
model) and scores every RSNA image with the adapted copy. The N images used to adapt are
flagged in the file and are left out of every score in lung.eval.run_fusion_rsna.

Usage:
    python -m lung.eval.run_adapted_rsna [--n 32]
Writes:
    reports/xray_cnn_adapted_rsna.csv   patient_id, p_adapted, used_for_adaptation
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lung.eval.common import REPORT_DIR, load_float_model, load_norm, load_rsna, normalise, predict_float


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32, help="number of unlabelled images to adapt on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lung.tta.bn_adapt import apply_bn_adaptation

    mu, sd = load_norm()
    man, X = load_rsna()
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(X), size=args.n, replace=False))

    model = load_float_model()
    adapted, collapsed = apply_bn_adaptation(model, normalise(X[idx], mu, sd))
    p = predict_float(adapted, X, mu, sd)

    used = np.zeros(len(X), dtype=bool)
    used[idx] = True
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"patient_id": man.patient_id.to_numpy(), "p_adapted": np.round(p, 6),
                  "used_for_adaptation": used}).to_csv(REPORT_DIR / "xray_cnn_adapted_rsna.csv", index=False)
    print(f"adapted on {args.n} images (collapse guard {'triggered' if collapsed else 'not triggered'}); "
          f"wrote reports/xray_cnn_adapted_rsna.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
