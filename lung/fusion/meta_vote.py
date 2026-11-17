"""Metadata vote for RSNA chest X-ray dataset.

Trains a logistic regression model on patient metadata (age, sex) using
5-fold cross-validation on the RSNA dataset.  View position is deliberately
excluded: it describes the imaging equipment, not the patient.

Features used:
    a   = age / 100
    a²  = (age / 100) **2
    sex_male = 1 if sex == 'M', else 0

When age is missing the age is filled with the training-fold median (strict
anti-leakage: the fill is computed on each fold's training split and applied to
the test split), BEFORE a and a^2 are formed, so that a^2 stays the square of a.
An unknown sex counts as 0.5. If no age is recorded at all, the vote is sex-only
and the JSON says so.

Output:
    reports/xray_pm_rsna.csv    -- image-level pM (out-of-fold)
    reports/xray_pm_rsna.json   -- AUROC summary

Usage:
    python -m lung.fusion.meta_vote
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lung.eval.common import meta_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "rsna_manifest.csv"
OUT_CSV = REPO_ROOT / "reports" / "xray_pm_rsna.csv"
OUT_JSON = REPO_ROOT / "reports" / "xray_pm_rsna.json"


META_COLUMNS = ["age_100", "age_100_sq", "sex_male"]
SEX_UNKNOWN = 0.5


def _age_fill(df: pd.DataFrame) -> float:
    """Median of the observed ages, 0.0 if none (the age terms are then constant)."""
    ages = pd.to_numeric(df["age"], errors="coerce").dropna()
    return float(ages.median()) if len(ages) else 0.0


def _lr() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000,
                              random_state=42)


def oof_metadata_vote(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold pM for every row of ``df`` (columns fold, label, age, sex). For fold k the
    age fill, the scaler and the regression come from the other folds only."""
    fold = df["fold"].to_numpy(dtype=int)
    y = df["label"].to_numpy(dtype=int)
    p = np.zeros(len(df))
    for f in sorted(set(fold.tolist())):
        tr, te = fold != f, fold == f
        X = meta_matrix(df, _age_fill(df[tr]))
        scaler = StandardScaler().fit(X[tr])
        p[te] = _lr().fit(scaler.transform(X[tr]), y[tr]).predict_proba(scaler.transform(X[te]))[:, 1]
    return p


def fit_metadata_vote(df: pd.DataFrame) -> dict:
    """Fit the final metadata vote on ALL rows of ``df`` and return plain numbers
    (age fill, scaler, coefficients) for reports/xray_fusion_params.json."""
    fill = _age_fill(df)
    X = meta_matrix(df, fill)
    scaler = StandardScaler().fit(X)
    clf = _lr().fit(scaler.transform(X), df["label"].to_numpy(dtype=int))
    return {
        "columns": META_COLUMNS,
        "age_fill_years": fill,
        "sex_unknown": SEX_UNKNOWN,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "age_observed_rows": int(pd.to_numeric(df["age"], errors="coerce").notna().sum()),
        "fitted_on": f"{len(df)} RSNA patients, all 5 folds",
    }


def run_meta_vote(
    manifest_csv: Path | None = None,
    out_csv: Path | None = None,
    out_json: Path | None = None,
) -> pd.DataFrame:
    """Run 5-fold metadata logistic regression on RSNA and return OOF DataFrame.

    Parameters
    ----------
    manifest_csv : path to rsna_manifest.csv (defaults to repo default)
    out_csv      : where to write xray_pm_rsna.csv
    out_json     : where to write the AUROC summary JSON

    Returns
    -------
    DataFrame with columns [index, patient_id, fold, label, pM]
    """
    manifest_csv = Path(manifest_csv or MANIFEST_PATH)
    out_csv = Path(out_csv or OUT_CSV)
    out_json = Path(out_json or OUT_JSON)

    df = pd.read_csv(manifest_csv)

    required = {"patient_id", "label", "fold", "sex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rsna_manifest.csv is missing columns: {missing}")

    # 'age' may be absent entirely or all-NaN; either way treat as NaN
    if "age" not in df.columns:
        df["age"] = np.nan

    folds = sorted(df["fold"].unique())
    if len(folds) != 5:
        raise ValueError(f"Expected 5 folds in RSNA manifest, found {folds}")

    p_oof = oof_metadata_vote(df)
    oof_rows = [{"index": int(i), "patient_id": str(r.patient_id), "fold": int(r.fold),
                 "label": int(r.label), "pM": float(p_oof[i])}
                for i, r in enumerate(df.itertuples())]

    df_oof = pd.DataFrame(oof_rows).sort_values("index").reset_index(drop=True)

    # --- Compute AUROC ---
    auroc_image = roc_auc_score(df_oof["label"], df_oof["pM"])

    # Patient-level: mean pM per patient
    patient_grp = df_oof.groupby("patient_id").agg(
        label=("label", "first"),
        pM=("pM", "mean"),
    )
    auroc_patient = roc_auc_score(patient_grp["label"], patient_grp["pM"])

    # --- Save ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_oof.to_csv(out_csv, index=False)

    summary = {
        "n_images": int(len(df_oof)),
        "n_patients": int(len(patient_grp)),
        "auroc_image_level": round(auroc_image, 4),
        "auroc_patient_level": round(auroc_patient, 4),
        "age_observed_rows": int(pd.to_numeric(df["age"], errors="coerce").notna().sum()),
        "inputs": ("age/100, (age/100)^2, sex" if pd.to_numeric(df["age"], errors="coerce").notna().any()
                   else "sex only: the manifest has no ages"),
        "view_position_used": False,
    }
    out_json.write_text(json.dumps(summary, indent=2))

    return df_oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="RSNA manifest CSV")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 1

    print(f"Running metadata vote on {args.manifest} ...")
    df_oof = run_meta_vote(args.manifest, args.out_csv, args.out_json)

    summary = json.loads(args.out_json.read_text())
    print(f"  Images: {summary['n_images']}  |  Patients: {summary['n_patients']}")
    print(f"  Image-level AUROC:   {summary['auroc_image_level']:.4f}")
    print(f"  Patient-level AUROC: {summary['auroc_patient_level']:.4f}")
    print(f"  Wrote {args.out_csv}")
    print(f"  Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
