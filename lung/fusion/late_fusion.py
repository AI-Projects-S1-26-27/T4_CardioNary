"""Late fusion: p = w1 * p_CNN + w2 * p_LR + w3 * p_M   (proposal 4.2, roadmap S2 and S5).

p_CNN is the CALIBRATED pneumonia probability (temperature scaling, lung.fusion.calibrate);
p_LR is the feature vote; p_M is the metadata vote (age and sex, never view position).

Rules
-----
* Weights are tried on a grid w1 + w2 + w3 = 1 in steps of 0.05 with w3 <= 0.3 and the one
  with the lowest log-loss is kept (lung.eval.common.choose_weights).
* The weights for fold k, and the temperature used for fold k, are fitted on the OTHER folds only.
* A missing vote gets weight 0 and the other weights are divided by their sum; the metadata
  weight is then clamped to 0.3 again, because dividing can push it above the cap.
* Every result is a FusedOutput("xray", ...) from core.contracts.

Kermany has no metadata, so there it is CNN + feature vote.

This module also holds ``xray_fusion``, the one function the dashboard's X-ray result uses. It
reads only reports/xray_fusion_params.json (plain numbers, no scikit-learn) and fits nothing.

Usage
-----
    python -m lung.fusion.late_fusion              # Kermany fusion, writes reports/xray_fused_kermany.*
    python -m lung.fusion.late_fusion --params     # (re)writes reports/xray_fusion_params.json
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.contracts import METADATA_WEIGHT_CAP, FusedOutput, PatientMetadata
from lung.eval.common import choose_weights, fuse, paired_auroc
from lung.eval.metrics import sensitivity_at_specificity
from lung.fusion.calibrate import apply_temperature, fit_temperature

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports"
PARAMS_JSON = REPORT_DIR / "xray_fusion_params.json"
RSNA_MANIFEST = REPO_ROOT / "data" / "manifests" / "rsna_manifest.csv"
SEX_UNKNOWN = 0.5


# ---------------------------------------------------------------------------
# Weights helpers
# ---------------------------------------------------------------------------
def renormalise(weights: Mapping[str, float], present, cap: float = METADATA_WEIGHT_CAP) -> dict[str, float]:
    """Keep the weights of the votes that are present, divide by their sum, then hold the
    metadata weight at the cap (the rest is shared in proportion). Falls back to equal
    weights if every present weight is 0."""
    names = [k for k in present]
    w = {k: float(weights.get(k, 0.0)) for k in names}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()} if total > 0 else {k: 1.0 / len(names) for k in names}
    if w.get("metadata", 0.0) > cap:
        rest = {k: v for k, v in w.items() if k != "metadata"}
        rest_total = sum(rest.values())
        w = ({k: (1.0 - cap) * (v / rest_total) for k, v in rest.items()} if rest_total > 0
             else {"cnn": 1.0 - cap})
        w["metadata"] = cap
        if "cnn" not in w:
            w["cnn"] = 1.0 - cap
    return w


# ---------------------------------------------------------------------------
# Out-of-fold fusion on a dataset (Kermany)
# ---------------------------------------------------------------------------
def fuse_dataset(
    df_oof: pd.DataFrame,
    df_plr: Optional[pd.DataFrame],
    df_pm: Optional[pd.DataFrame],
    dataset_name: str,
    out_csv: Path,
    out_json: Path,
) -> list[FusedOutput]:
    """Out-of-fold late fusion. The three frames must describe the same images in the same
    order (checked on label and patient_id); df_plr / df_pm may be None."""
    df_oof = df_oof.sort_values("index").reset_index(drop=True) if "index" in df_oof.columns \
        else df_oof.reset_index(drop=True)
    n = len(df_oof)
    y = df_oof["label"].to_numpy(int)
    fold = df_oof["fold"].to_numpy(int)
    pid = df_oof["patient_id"].astype(str).to_numpy()
    p_raw = df_oof["p_pneumonia"].to_numpy(float)

    votes_raw = {"cnn": p_raw}
    for name, df, col in (("features", df_plr, "pLR"), ("metadata", df_pm, "pM")):
        if df is None:
            continue
        df = df.reset_index(drop=True)
        if len(df) != n or not (df["label"].to_numpy(int) == y).all() \
                or not (df["patient_id"].astype(str).to_numpy() == pid).all():
            raise ValueError(f"{name} vote file is not row-aligned with the CNN out-of-fold file; "
                             f"regenerate both from the same cache")
        votes_raw[name] = df[col].to_numpy(float)

    p_cal = np.zeros(n)
    p_fused = np.zeros(n)
    w_cols = {k: np.zeros(n) for k in ("cnn", "features", "metadata")}
    per_fold, fused_outputs = [], []

    for f in sorted(set(fold.tolist())):
        te, tr = fold == f, fold != f
        # Temperature from the other folds only, applied to this fold
        T = fit_temperature(p_raw[tr], y[tr])
        votes_tr = {k: v[tr].copy() for k, v in votes_raw.items()}
        votes_te = {k: v[te].copy() for k, v in votes_raw.items()}
        votes_tr["cnn"] = apply_temperature(votes_tr["cnn"], T)
        votes_te["cnn"] = apply_temperature(votes_te["cnn"], T)
        p_cal[te] = votes_te["cnn"]

        weights = renormalise(choose_weights(votes_tr, y[tr], units=20, metadata_cap=METADATA_WEIGHT_CAP),
                              votes_te.keys())
        p_fused[te] = fuse(votes_te, weights)
        for k in w_cols:
            w_cols[k][te] = weights.get(k, 0.0)
        per_fold.append({"fold": int(f), "temperature": float(T), "weights": weights})

    for i in range(n):
        votes_i = {k: float(np.clip(v[i], 0.0, 1.0)) for k, v in votes_raw.items()}
        votes_i["cnn"] = float(np.clip(p_cal[i], 0.0, 1.0))
        weights_i = {k: float(w_cols[k][i]) for k in votes_i}
        fused_outputs.append(FusedOutput("xray", float(np.clip(p_fused[i], 0.0, 1.0)), votes_i, weights_i))

    df_out = pd.DataFrame({
        "index": np.arange(n), "patient_id": pid, "fold": fold, "label": y,
        "p_cnn_calibrated": p_cal, "p_fused": p_fused,
        "w_cnn": w_cols["cnn"], "w_features": w_cols["features"], "w_metadata": w_cols["metadata"],
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    patients = df_out.groupby("patient_id").agg(label=("label", "first"), p_fused=("p_fused", "mean"),
                                                p_cnn=("p_cnn_calibrated", "mean"))
    summary = {
        "dataset": dataset_name,
        "votes_used": list(votes_raw),
        "n_images": int(n),
        "n_patients": int(len(patients)),
        "calibrated_cnn": {"auroc": float(roc_auc_score(y, p_cal)),
                           "sensitivity_at_90_specificity": sensitivity_at_specificity(y, p_cal, 0.90)},
        "fused": {"auroc": float(roc_auc_score(y, p_fused)),
                  "sensitivity_at_90_specificity": sensitivity_at_specificity(y, p_fused, 0.90)},
        "cnn_vs_fused": paired_auroc(y, p_cal, p_fused),
        "auroc_image_level": round(float(roc_auc_score(y, p_fused)), 4),
        "auroc_patient_level": round(float(roc_auc_score(patients["label"], patients["p_fused"])), 4),
        "per_fold": per_fold,
        "note": "temperature and weights for each fold are fitted on the other four folds only; "
                "bootstrap: 1,000 resamples of images, seed 0",
    }
    out_json.write_text(json.dumps(summary, indent=2))
    return fused_outputs


# ---------------------------------------------------------------------------
# Fitted numbers for the dashboard
# ---------------------------------------------------------------------------
def build_fusion_params(
    oof_csv: Path = REPORT_DIR / "xray_oof_kermany.csv",
    features_csv: Path = REPORT_DIR / "xray_features_kermany.csv",
    manifest_csv: Path = RSNA_MANIFEST,
    fusion_json: Path = REPORT_DIR / "xray_fusion.json",
    out_path: Path = PARAMS_JSON,
) -> dict:
    """Fit T, the two votes and the weights and save them as plain numbers.

    T          Kermany out-of-fold CNN probabilities, all folds (the dashboard fits nothing).
    features   logistic regression on the 12 characteristics, all Kermany images.
    metadata   logistic regression on age and sex, all RSNA patients.
    weights    ``app_weights`` of reports/xray_fusion.json (chosen on RSNA out-of-fold votes) when
               that file exists; otherwise cnn + features chosen on Kermany out-of-fold votes and
               a metadata weight of 0.
    """
    from lung.fusion.meta_vote import fit_metadata_vote
    from lung.fusion.votes import fit_feature_vote

    oof = pd.read_csv(oof_csv).sort_values("index").reset_index(drop=True)
    T = fit_temperature(oof["p_pneumonia"].to_numpy(float), oof["label"].to_numpy(int))

    feats = pd.read_csv(features_csv)
    feature_vote = fit_feature_vote(feats)

    metadata_vote = None
    if Path(manifest_csv).exists():
        metadata_vote = fit_metadata_vote(pd.read_csv(manifest_csv, dtype={"patient_id": str}))

    weights, source = None, None
    if Path(fusion_json).exists():
        aw = json.loads(Path(fusion_json).read_text()).get("app_weights")
        if aw:
            weights, source = {k: float(v) for k, v in aw.items()}, \
                "app_weights of reports/xray_fusion.json (chosen on RSNA out-of-fold votes)"
    if weights is None:
        plr = pd.read_csv(REPORT_DIR / "xray_plr_kermany.csv")
        p_cal = apply_temperature(oof["p_pneumonia"].to_numpy(float), T)
        weights = choose_weights({"cnn": p_cal, "features": plr["pLR"].to_numpy(float)},
                                 oof["label"].to_numpy(int))
        weights["metadata"] = 0.0
        source = "chosen on Kermany out-of-fold votes (no RSNA run was available)"
    weights = {k: float(weights.get(k, 0.0)) for k in ("cnn", "features", "metadata")}
    if metadata_vote is None:
        weights = renormalise(weights, ["cnn", "features"])
        weights["metadata"] = 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9 and weights["metadata"] <= METADATA_WEIGHT_CAP + 1e-9

    params = {
        "T": float(T),
        "weights": weights,
        "weights_source": source,
        "feature_vote": feature_vote,
        "metadata_vote": metadata_vote,
        "notes": "p_cnn given to xray_fusion is the calibrated pneumonia probability. The metadata vote "
                 "never uses view position. Everything here is a plain number; nothing is fitted in the app.",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(params, indent=2))
    load_fusion_params.cache_clear()
    return params


@lru_cache(maxsize=4)
def load_fusion_params(path: str | Path = PARAMS_JSON) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; create it with: python -m lung.fusion.late_fusion --params")
    return json.loads(p.read_text())


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _linear_vote(x: list[float], vote: dict) -> float:
    z = float(vote["intercept"])
    for xi, mu, sc, c in zip(x, vote["scaler_mean"], vote["scaler_scale"], vote["coef"]):
        z += c * ((xi - mu) / (sc if sc else 1.0))
    return _sigmoid(z)


def feature_vote_probability(features: Optional[Mapping[str, float]], vote: dict) -> Optional[float]:
    """p_LR, or None when the characteristics are missing altogether. A single missing or
    non-finite value is replaced by its filling value (the training median)."""
    if not features:
        return None
    x, seen = [], 0
    for col, fill in zip(vote["columns"], vote["fill"]):
        v = features.get(col)
        try:
            ok = v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            ok = False
        seen += ok
        x.append(float(v) if ok else float(fill))
    return _linear_vote(x, vote) if seen else None


def metadata_vote_probability(metadata: Optional[PatientMetadata], vote: Optional[dict]) -> Optional[float]:
    """p_M, or None when neither age nor sex was entered (or the vote was not fitted)."""
    if vote is None or metadata is None or (metadata.age_years is None and metadata.sex is None):
        return None
    age = float(vote["age_fill_years"] if metadata.age_years is None else metadata.age_years) / 100.0
    sex = {"male": 1.0, "female": 0.0}.get(metadata.sex, vote.get("sex_unknown", SEX_UNKNOWN))
    return _linear_vote([age, age ** 2, sex], vote)


def xray_fusion(
    p_cnn: float,
    features: Optional[Mapping[str, float]] = None,
    metadata: Optional[PatientMetadata] = None,
    params: Optional[dict] = None,
) -> FusedOutput:
    """Late fusion for one X-ray, from reports/xray_fusion_params.json only.

    p_cnn     calibrated pneumonia probability of the CNN
    features  the 12 characteristics (lung.model.features.characteristics), or None
    metadata  PatientMetadata (age and sex are used; symptoms are shown, never used)
    """
    if not 0.0 <= float(p_cnn) <= 1.0 or not math.isfinite(float(p_cnn)):
        raise ValueError(f"p_cnn must be a probability, got {p_cnn!r}")
    params = params if params is not None else load_fusion_params(str(PARAMS_JSON))

    votes = {"cnn": float(p_cnn)}
    p_lr = feature_vote_probability(features, params["feature_vote"])
    if p_lr is not None:
        votes["features"] = p_lr
    p_m = metadata_vote_probability(metadata, params.get("metadata_vote"))
    if p_m is not None:
        votes["metadata"] = p_m

    weights = renormalise(params["weights"], votes.keys())
    p = sum(weights[k] * votes[k] for k in votes)
    return FusedOutput("xray", float(min(max(p, 0.0), 1.0)), votes, weights)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", action="store_true",
                        help="write reports/xray_fusion_params.json instead of the Kermany fusion")
    args = parser.parse_args()

    if args.params:
        p = build_fusion_params()
        print(f"T = {p['T']:.4f}; weights = {p['weights']} ({p['weights_source']})")
        mv = p["metadata_vote"]
        print("metadata vote:", "not fitted (no RSNA manifest)" if mv is None
              else f"{mv['age_observed_rows']} ages observed")
        print(f"wrote {PARAMS_JSON}")
        return 0

    oof_path, plr_path = REPORT_DIR / "xray_oof_kermany.csv", REPORT_DIR / "xray_plr_kermany.csv"
    for p in (oof_path, plr_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    print("Fusing Kermany: calibrated CNN + feature vote (no metadata in Kermany) ...")
    fo = fuse_dataset(pd.read_csv(oof_path), pd.read_csv(plr_path), None, "kermany",
                      REPORT_DIR / "xray_fused_kermany.csv", REPORT_DIR / "xray_fused_kermany.json")
    s = json.loads((REPORT_DIR / "xray_fused_kermany.json").read_text())
    print(f"  CNN AUROC {s['calibrated_cnn']['auroc']:.4f} -> fused {s['fused']['auroc']:.4f}; "
          f"{len(fo)} FusedOutput objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
