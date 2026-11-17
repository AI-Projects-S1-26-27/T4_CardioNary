"""S2: the two fusion comparisons, plus CNN calibration, written to reports/xray_fusion.json.

Votes: CNN (final model), feature vote (logistic regression on the 12 image characteristics,
trained on Kermany), metadata vote (logistic regression on age and sex, trained on RSNA folds).

1. Generalisation (no RSNA label used to build anything): plain CNN vs CNN + feature vote,
   weights chosen on Kermany out-of-fold predictions, scored on RSNA.
2. Metadata gain (within RSNA's 5 folds): CNN + feature vote, with and without the metadata
   vote (weight <= 0.3). For each fold the weights are chosen on the other four folds only,
   and the metadata vote is out-of-fold.

Needs (run these first, see the step list):
    reports/xray_cnn_rsna.csv, reports/xray_features_rsna.csv
    reports/xray_oof_kermany.csv, reports/xray_features_kermany.csv, reports/xray_plr_kermany.csv

Usage:
    python -m lung.eval.run_fusion_rsna
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lung.eval.common import REPORT_DIR, choose_weights, fuse, paired_auroc
from lung.eval.metrics import expected_calibration_error, sensitivity_at_specificity
from lung.fusion.calibrate import apply_temperature, fit_temperature
from lung.fusion.meta_vote import oof_metadata_vote
from lung.fusion.votes import FEATURE_COLUMNS

HINTS = {
    "xray_cnn_rsna.csv": "python -m lung.eval.run_rsna",
    "xray_features_rsna.csv": "python -m lung.data.pack_rsna_cache && python -m lung.model.features "
                              "--cache data/cache/rsna_96.npz --manifest data/manifests/rsna_manifest.csv "
                              "--output reports/xray_features_rsna.csv",
    "xray_features_kermany.csv": "python -m lung.model.features",
    "xray_plr_kermany.csv": "python -m lung.fusion.votes",
    "xray_oof_kermany.csv": "python -m lung.model.cv_oof   (about 40 minutes)",
}


def read(name: str, **kw) -> pd.DataFrame:
    p = REPORT_DIR / name
    if not p.exists():
        raise SystemExit(f"missing {p}\n  create it with: {HINTS[name]}")
    return pd.read_csv(p, **kw)


def lr() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)


def summary(y, p) -> dict:
    return {"auroc": float(roc_auc_score(y, p)),
            "sensitivity_at_90_specificity": sensitivity_at_specificity(y, p, 0.90)}


def compute_generalisation_comparison(
    y: np.ndarray,
    p_cnn: np.ndarray,
    p_feat: np.ndarray,
    w_k: dict,
    p_adapted: np.ndarray | None = None,
) -> dict:
    """Compare plain CNN, CNN + feature vote, and optionally adapted CNN (full CardioNary branch)."""
    fused1 = fuse({"cnn": p_cnn, "features": p_feat}, w_k)
    gen = paired_auroc(y, p_cnn, fused1)
    res = {
        "weights_chosen_on": "Kermany out-of-fold predictions",
        "weights": w_k,
        "plain_cnn_vs_cnn_plus_features": gen,
        "feature_vote_alone": summary(y, p_feat),
        "plain_cnn": summary(y, p_cnn),
        "cnn_plus_features": summary(y, fused1),
        "target": "AUROC gain >= +0.03",
        "gain": gen["difference_b_minus_a"]["estimate"],
        "meets_target": bool(gen["difference_b_minus_a"]["estimate"] >= 0.03),
    }

    # Add adapted model and the full CardioNary branch (adapted CNN + features)
    if p_adapted is not None:
        p_adapt = np.asarray(p_adapted, dtype=float)
        fused_adapt = fuse({"cnn": p_adapt, "features": p_feat}, w_k)
        gen_adapt = paired_auroc(y, p_cnn, p_adapt)
        gen_full = paired_auroc(y, p_cnn, fused_adapt)
        gen_features_vs_full = paired_auroc(y, fused1, fused_adapt)
        res.update({
            "adapted_cnn": summary(y, p_adapt),
            "adapted_cnn_plus_features": summary(y, fused_adapt),
            "plain_cnn_vs_adapted_cnn": gen_adapt,
            "plain_cnn_vs_adapted_cnn_plus_features": gen_full,
            "cnn_plus_features_vs_adapted_cnn_plus_features": gen_features_vs_full,
            "adapted_gain": gen_adapt["difference_b_minus_a"]["estimate"],
            "full_cardionary_gain": gen_full["difference_b_minus_a"]["estimate"],
        })

    return res


def main() -> int:
    # ---- inputs ----------------------------------------------------------
    cnn = read("xray_cnn_rsna.csv", dtype={"patient_id": str})
    ft = read("xray_features_rsna.csv", dtype={"patient_id": str})
    rs = cnn.merge(ft[["patient_id"] + FEATURE_COLUMNS], on="patient_id", validate="one_to_one")
    if len(rs) != len(cnn):
        raise SystemExit("RSNA CNN file and RSNA feature file do not cover the same patients")
    oof = read("xray_oof_kermany.csv").sort_values("index").reset_index(drop=True)
    plr = read("xray_plr_kermany.csv")
    fk = read("xray_features_kermany.csv")
    if not (len(oof) == len(plr) == len(fk)
            and (oof.label.to_numpy() == plr.label.to_numpy()).all()
            and (oof.patient_id.astype(str).to_numpy() == plr.patient_id.astype(str).to_numpy()).all()):
        raise SystemExit("Kermany out-of-fold, vote and feature files are not row-aligned; regenerate them "
                         "from the same cache")

    y, fold = rs.label.to_numpy(int), rs.fold.to_numpy(int)
    yk, pk_raw = oof.label.to_numpy(int), oof.p_pneumonia.to_numpy(float)

    # ---- calibration of the CNN (Kermany out-of-fold) --------------------
    # Every vote below uses the CALIBRATED CNN probability, as lung.predict / xray_fusion do.
    T = fit_temperature(pk_raw, yk)
    calibration = {
        "temperature": float(T),
        "fitted_on": "Kermany out-of-fold CNN probabilities, all 5 folds",
        "ece_before": expected_calibration_error(yk, pk_raw, n_bins=15),
        "ece_after": expected_calibration_error(yk, apply_temperature(pk_raw, T), n_bins=15),
        "target": "ece_after <= 0.05",
    }
    pk = apply_temperature(pk_raw, T)
    p_cnn_raw = rs.p_float.to_numpy(float)
    p_cnn = apply_temperature(p_cnn_raw, T)
    calibration["rsna_ece_before"] = expected_calibration_error(y, p_cnn_raw, n_bins=15)
    calibration["rsna_ece_after"] = expected_calibration_error(y, p_cnn, n_bins=15)

    # Adapted model predictions (lung.eval.run_adapted_rsna). The images it adapted on are never scored.
    p_adapted, keep = None, np.ones(len(rs), dtype=bool)
    if "p_adapted" in rs.columns:
        p_adapted = apply_temperature(rs["p_adapted"].to_numpy(float), T)
    elif (REPORT_DIR / "xray_cnn_adapted_rsna.csv").exists():
        adapt_df = read("xray_cnn_adapted_rsna.csv", dtype={"patient_id": str})
        m_adapt = rs[["patient_id"]].merge(adapt_df, on="patient_id", how="left")
        if m_adapt["p_adapted"].notna().all():
            p_adapted = apply_temperature(m_adapt["p_adapted"].to_numpy(float), T)
            keep = ~m_adapt["used_for_adaptation"].fillna(False).to_numpy(bool)
        else:
            print("[warn] adapted file does not cover every RSNA patient; adapted model left out")

    # ---- feature vote trained on ALL of Kermany, applied to RSNA ---------
    sc = StandardScaler().fit(fk[FEATURE_COLUMNS])
    clf = lr().fit(sc.transform(fk[FEATURE_COLUMNS]), fk.label)
    p_feat = clf.predict_proba(sc.transform(rs[FEATURE_COLUMNS]))[:, 1]

    # ---- 1. generalisation -----------------------------------------------
    w_k = choose_weights({"cnn": pk, "features": plr.pLR.to_numpy(float)}, yk)
    generalisation = compute_generalisation_comparison(
        y=y[keep],
        p_cnn=p_cnn[keep],
        p_feat=p_feat[keep],
        w_k=w_k,
        p_adapted=None if p_adapted is None else p_adapted[keep],
    )
    generalisation["n_scored"] = int(keep.sum())
    generalisation["n_excluded_used_for_adaptation"] = int((~keep).sum())

    # ---- 2. metadata gain within RSNA folds ------------------------------
    folds = sorted(set(fold.tolist()))
    rsna_age = pd.to_numeric(rs["age"], errors="coerce")
    age_observed_rows = int(rsna_age.notna().sum())
    # Same definition as lung.fusion.meta_vote (training-fold age fill and scaler).
    p_meta = oof_metadata_vote(rs)

    votes2 = {"cnn": p_cnn, "features": p_feat}
    votes3 = {**votes2, "metadata": p_meta}
    f2, f3, per_fold = np.zeros(len(y)), np.zeros(len(y)), []
    for k in folds:
        tr, te = fold != k, fold == k
        w2 = choose_weights({a: v[tr] for a, v in votes2.items()}, y[tr])
        w3 = choose_weights({a: v[tr] for a, v in votes3.items()}, y[tr])
        f2[te] = fuse({a: v[te] for a, v in votes2.items()}, w2)
        f3[te] = fuse({a: v[te] for a, v in votes3.items()}, w3)
        per_fold.append({"fold": int(k), "weights_without_metadata": w2, "weights_with_metadata": w3})
    meta = paired_auroc(y, f2, f3)

    # The same two fusions with the adapted CNN (roadmap section 5: each version with and without
    # adaptation). Weights are rechosen on the training folds; the images used to adapt are not scored.
    adapted_versions = None
    if p_adapted is not None:
        a2, a3 = np.zeros(len(y)), np.zeros(len(y))
        av2, av3 = {"cnn": p_adapted, "features": p_feat}, {"cnn": p_adapted, "features": p_feat, "metadata": p_meta}
        for k in folds:
            tr, te = fold != k, fold == k
            a2[te] = fuse({a: v[te] for a, v in av2.items()},
                          choose_weights({a: v[tr] for a, v in av2.items()}, y[tr]))
            a3[te] = fuse({a: v[te] for a, v in av3.items()},
                          choose_weights({a: v[tr] for a, v in av3.items()}, y[tr]))
        adapted_versions = {
            "scored": f"{int(keep.sum())} patients (the {int((~keep).sum())} used to adapt are left out)",
            "adapted_cnn_plus_features": summary(y[keep], a2[keep]),
            "adapted_cnn_plus_features_plus_metadata": summary(y[keep], a3[keep]),
            "metadata_gain_with_adapted_cnn": paired_auroc(y[keep], a2[keep], a3[keep]),
            "plain_cnn_vs_full_branch_with_metadata": paired_auroc(y[keep], p_cnn[keep], a3[keep]),
            "plain_cnn_plus_features_plus_metadata_vs_adapted": paired_auroc(y[keep], f3[keep], a3[keep]),
        }
    metadata_gain = {
        "weights_chosen_on": "the other four RSNA folds, rechosen for each fold",
        "age_observed_rows": age_observed_rows,
        "age_note": (
            "No RSNA ages were available; age features are constant after imputation, so the metadata vote uses sex only."
            if age_observed_rows == 0 else
            "Missing ages are imputed with each training fold's median."
        ),
        "per_fold": per_fold,
        "cnn_plus_features_vs_plus_metadata": meta,
        "metadata_vote_alone": summary(y, p_meta),
        "without_metadata": summary(y, f2),
        "with_metadata": summary(y, f3),
        "mean_metadata_weight": float(np.mean([f["weights_with_metadata"]["metadata"] for f in per_fold])),
        "target": "AUROC gain >= +0.02 with w3 <= 0.3",
        "gain": meta["difference_b_minus_a"]["estimate"],
        "meets_target": bool(meta["difference_b_minus_a"]["estimate"] >= 0.02),
        "with_adapted_cnn": adapted_versions,
        "note": "metadata vote is trained on adults (RSNA) while the CNN is trained on children (Kermany); "
                "view position is never an input",
    }

    # weights for the offline tool (a demo setting, chosen on all RSNA out-of-fold votes; NOT an evaluation)
    app_weights = choose_weights(votes3, y)

    out = {"calibration_cnn_kermany": calibration, "generalisation": generalisation,
           "metadata_gain": metadata_gain, "app_weights": app_weights,
           "bootstrap": "1,000 resamples of cases (one image per RSNA patient), seed 0, 95% percentile interval"}
    (REPORT_DIR / "xray_fusion.json").write_text(json.dumps(out, indent=2))

    g, m = generalisation, metadata_gain
    print(f"calibration: T={T:.3f}, ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}")
    print(f"1. generalisation: CNN {g['plain_cnn']['auroc']:.4f} -> CNN+features "
          f"{g['cnn_plus_features']['auroc']:.4f}  gain {g['gain']:+.4f} "
          f"{g['plain_cnn_vs_cnn_plus_features']['difference_b_minus_a']['ci95']}  weights {w_k}")
    if "adapted_cnn_plus_features" in g:
        print(f"   adapted CNN: {g['adapted_cnn']['auroc']:.4f} (gain {g['adapted_gain']:+.4f}) -> "
              f"adapted+features (full branch): {g['adapted_cnn_plus_features']['auroc']:.4f} "
              f"(gain {g['full_cardionary_gain']:+.4f})")
    print(f"2. metadata gain:  {m['without_metadata']['auroc']:.4f} -> {m['with_metadata']['auroc']:.4f}  "
          f"gain {m['gain']:+.4f} {meta['difference_b_minus_a']['ci95']}  mean w3 {m['mean_metadata_weight']:.2f}")
    if adapted_versions:
        av = adapted_versions
        print(f"   adapted CNN + features + metadata: {av['adapted_cnn_plus_features_plus_metadata']['auroc']:.4f}; "
              f"metadata gain with adapted CNN {av['metadata_gain_with_adapted_cnn']['difference_b_minus_a']['estimate']:+.4f}")
    print("wrote reports/xray_fusion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
