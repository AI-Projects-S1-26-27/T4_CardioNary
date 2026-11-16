"""Late fusion of heart sound CNN probabilities, feature vote, and metadata vote.

Task 5: Combines the votes into p_h = w1*p_CNN + w2*p_LR + w3*p_M (w3 <= 0.3).
Runs Comparison A (generalisation to PhysioNet) and Comparison B (metadata gain).

Task 6: Adds the adapted model (adapt_bn) to Comparison A:
- plain CNN
- adapted CNN
- adapted CNN plus feature vote
"""
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Generator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from core.contracts import FusedOutput, PatientMetadata
from cardio.eval.intervals import metric_intervals
from cardio.eval.metrics import classification_metrics
from cardio.eval.real_world import adapt_bn
from cardio.fusion.calibrate import calibrate_probabilities
from cardio.fusion.votes import AGE_MAP, new_vote, run_feature_vote, run_metadata_vote
from cardio.model.final_model import CLASSES, FinalModel


def weight_grid(use_metadata: bool, step: float = 0.05, w3_max: float = 0.3) -> Generator[Tuple[float, float, float], None, None]:
    """Generate valid late fusion weight combinations summing to 1."""
    n = round(1 / step)
    for i in range(n + 1):
        for k in range(n + 1 - i):
            w1, w3 = round(i * step, 4), round(k * step, 4)
            if w3 > w3_max + 1e-9 or (k > 0 and not use_metadata):
                continue
            w2 = round(1.0 - w1 - w3, 4)
            if w2 < -1e-9:
                continue
            yield round(w1, 2), round(w2, 2), round(w3, 2)


def best_weights(
    y: np.ndarray,
    p_cnn: np.ndarray,
    p_lr: np.ndarray,
    p_m: Optional[np.ndarray] = None,
    step: float = 0.05,
    w3_max: float = 0.3,
) -> Tuple[float, float, float]:
    """Find weights minimizing log-loss."""
    use_meta = p_m is not None

    def loss(w: Tuple[float, float, float]) -> float:
        p = w[0] * p_cnn + w[1] * p_lr + (w[2] * p_m if use_meta else 0.0)
        return float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6)))

    return min(weight_grid(use_meta, step=step, w3_max=w3_max), key=loss)


def generate_physionet_cnn_probabilities() -> pd.DataFrame:
    """Run FinalModel over PhysioNet cached features if not already generated."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "heart_cnn_physionet.csv"

    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Extracting CNN probabilities for PhysioNet from cached features...")
    model = FinalModel()
    rows = []
    with open("data/manifests/circor_manifest.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            with np.load(r["cache_path"]) as cache:
                feats = cache["features"]
                if len(feats) == 0:
                    continue
                p = np.asarray(model.float_model(model.normalise(feats), training=False))
                p = p.reshape(len(feats), -1).mean(axis=0)
                rows.append({
                    "patient_id": r["patient_id"],
                    "patient_id": str(r["patient_id"]),
                    "path": r["path"],
                    **{f"p_{k}": float(v) for k, v in zip(CLASSES, p)},
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} PhysioNet CNN probability rows to {out_csv}")
    return df


def run_physionet_adaptation(T: float) -> pd.DataFrame:
    """Task 6: Split PhysioNet into two random halves (seed 0), adapt with 64 clips, and measure."""
    print("\nRunning Task 6: Test-time batch-norm adaptation (adapt_bn)...")
    manifest_path = Path("data/manifests/circor_manifest.csv")
    rows = []
    with open(manifest_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Split patients into 2 random halves with seed 0
    patients = sorted({r["patient_id"] for r in rows})
    order = np.random.default_rng(0).permutation(len(patients))
    half = len(patients) // 2
    halves = [set(patients[i] for i in order[:half]), set(patients[i] for i in order[half:])]

    model = FinalModel()
    N_ADAPT = 64
    adapted_predictions = {}

    for i, adapt_on in enumerate(halves):
        measure_on = halves[1 - i]
        pool = [r for r in rows if r["patient_id"] in adapt_on]
        pick = np.random.default_rng(0).permutation(len(pool))[:N_ADAPT]
        pool_adapt = [pool[idx] for idx in sorted(pick)]

        adapt_feats = []
        for r in pool_adapt:
            with np.load(r["cache_path"]) as c:
                f = c["features"]
                if len(f) > 0:
                    adapt_feats.append(f)
        windows = np.concatenate(adapt_feats)
        adapted_model = adapt_bn(windows, model)

        measure_rows = [r for r in rows if r["patient_id"] in measure_on]
        for r in measure_rows:
            with np.load(r["cache_path"]) as c:
                f = c["features"]
                if len(f) == 0:
                    continue
                p = np.asarray(adapted_model(model.normalise(f), training=False)).reshape(len(f), -1).mean(0)
                adapted_predictions[r["path"]] = p

    adapted_rows = []
    for r in rows:
        if r["path"] in adapted_predictions:
            p = adapted_predictions[r["path"]]
            adapted_rows.append({
                "patient_id": str(r["patient_id"]),
                "path": r["path"],
                **{f"p_{k}": float(v) for k, v in zip(CLASSES, p)},
            })

    df_adapt = pd.DataFrame(adapted_rows)
    P_adapt = df_adapt.filter(like="p_").to_numpy(dtype=float)
    P_adapt = P_adapt / P_adapt.sum(axis=1, keepdims=True)
    P_adapt_cal = calibrate_probabilities(P_adapt, T)
    df_adapt["p_cnn_adapt"] = 1.0 - P_adapt_cal[:, 0]

    patient_adapt = df_adapt.groupby("patient_id")["p_cnn_adapt"].mean().reset_index()
    return patient_adapt


def run_late_fusion():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load calibration temperature T
    cal_file = reports_dir / "heart_calibration.json"
    if not cal_file.exists():
        raise FileNotFoundError("Missing heart_calibration.json. Run cardio.fusion.calibrate first.")
    with open(cal_file, "r", encoding="utf-8") as fh:
        T = json.load(fh)["final_T"]

    # 2. Fit weights on Yaseen
    oof_yaseen = pd.read_csv(reports_dir / "heart_oof_yaseen.csv")
    P_y = oof_yaseen.filter(like="p_").to_numpy(dtype=float)
    P_y = P_y / P_y.sum(axis=1, keepdims=True)
    P_y_cal = calibrate_probabilities(P_y, T)
    oof_yaseen["p_cnn"] = 1.0 - P_y_cal[:, 0]

    plr_yaseen = pd.read_csv(reports_dir / "heart_plr_yaseen.csv")

    # Join on patient_id (drops the 10 short clips, exactly per task guide)
    yaseen_merged = pd.merge(
        oof_yaseen[["patient_id", "fold", "label", "p_cnn"]],
        plr_yaseen[["patient_id", "p_lr"]],
        on="patient_id",
    )

    y_yaseen = (yaseen_merged["label"] != 0).astype(int).to_numpy()
    p_cnn_y = yaseen_merged["p_cnn"].to_numpy(dtype=float)
    p_lr_y = yaseen_merged["p_lr"].to_numpy(dtype=float)

    w_yaseen = best_weights(y_yaseen, p_cnn_y, p_lr_y)
    print(f"Fitted Yaseen weights (w1_cnn, w2_lr, w3_m): {w_yaseen}")

    # 3. Load PhysioNet data
    df_cnn_phys = generate_physionet_cnn_probabilities()
    df_cnn_phys["patient_id"] = df_cnn_phys["patient_id"].astype(str)
    P_phys = df_cnn_phys.filter(like="p_").to_numpy(dtype=float)
    P_phys = P_phys / P_phys.sum(axis=1, keepdims=True)
    P_phys_cal = calibrate_probabilities(P_phys, T)
    df_cnn_phys["p_cnn"] = 1.0 - P_phys_cal[:, 0]
    patient_cnn = df_cnn_phys.groupby("patient_id")["p_cnn"].mean().reset_index()

    # Fit feature model on all Yaseen and predict for PhysioNet
    feats_y = pd.read_csv(reports_dir / "heart_features_yaseen.csv")
    feature_cols = [c for c in feats_y.columns if c not in ["patient_id", "fold", "label"]]
    full_feat_model = new_vote().fit(
        feats_y[feature_cols].to_numpy(dtype=float),
        (feats_y["label"] != 0).astype(int).to_numpy(),
    )

    feats_phys = pd.read_csv(reports_dir / "heart_features_physionet.csv")
    feats_phys["patient_id"] = feats_phys["patient_id"].astype(str)
    patient_feats = feats_phys.groupby("patient_id")[feature_cols].mean().reset_index()
    patient_feats["p_lr"] = full_feat_model.predict_proba(
        patient_feats[feature_cols].to_numpy(dtype=float)
    )[:, 1]

    # Load metadata vote probabilities and patient folds
    df_pm_phys = pd.read_csv(reports_dir / "heart_pm_physionet.csv")
    df_pm_phys["patient_id"] = df_pm_phys["patient_id"].astype(str)

    # Task 6: Adapted CNN probabilities
    patient_cnn_adapt = run_physionet_adaptation(T)

    # Merge on patient_id (816 patients)
    merged = pd.merge(df_pm_phys, patient_cnn[["patient_id", "p_cnn"]], on="patient_id")
    merged = pd.merge(merged, patient_feats[["patient_id", "p_lr"]], on="patient_id")
    merged = pd.merge(merged, patient_cnn_adapt[["patient_id", "p_cnn_adapt"]], on="patient_id")

    y_phys = merged["label"].to_numpy(dtype=int)
    groups_phys = merged["patient_id"].to_numpy()
    p_cnn_phys = merged["p_cnn"].to_numpy(dtype=float)
    p_lr_phys = merged["p_lr"].to_numpy(dtype=float)
    p_m_phys = merged["p_m"].to_numpy(dtype=float)
    p_cnn_adapt_phys = merged["p_cnn_adapt"].to_numpy(dtype=float)
    folds_phys = merged["fold"].to_numpy(dtype=int)

    # 4. Comparison A: Generalisation
    p_compA_plain = p_cnn_phys
    p_compA_fused = w_yaseen[0] * p_cnn_phys + w_yaseen[1] * p_lr_phys
    p_compA_adapted = p_cnn_adapt_phys
    p_compA_adapted_fused = w_yaseen[0] * p_cnn_adapt_phys + w_yaseen[1] * p_lr_phys

    metrics_A_plain = metric_intervals(y_phys, p_compA_plain, groups_phys, ["macro_f1", "auroc"])
    metrics_A_fused = metric_intervals(y_phys, p_compA_fused, groups_phys, ["macro_f1", "auroc"])
    metrics_A_adapted = metric_intervals(y_phys, p_compA_adapted, groups_phys, ["macro_f1", "auroc"])
    metrics_A_adapted_fused = metric_intervals(y_phys, p_compA_adapted_fused, groups_phys, ["macro_f1", "auroc"])

    print("\n=== Comparison A: Generalisation (Yaseen -> PhysioNet) ===")
    print(f"Plain CNN AUROC: {metrics_A_plain['auroc']['estimate']:.4f} [{metrics_A_plain['auroc']['lower_95']:.4f}, {metrics_A_plain['auroc']['upper_95']:.4f}], F1: {metrics_A_plain['macro_f1']['estimate']:.4f}")
    print(f"Fused CNN+LR AUROC: {metrics_A_fused['auroc']['estimate']:.4f} [{metrics_A_fused['auroc']['lower_95']:.4f}, {metrics_A_fused['auroc']['upper_95']:.4f}], F1: {metrics_A_fused['macro_f1']['estimate']:.4f}")
    print(f"Adapted CNN AUROC: {metrics_A_adapted['auroc']['estimate']:.4f} [{metrics_A_adapted['auroc']['lower_95']:.4f}, {metrics_A_adapted['auroc']['upper_95']:.4f}], F1: {metrics_A_adapted['macro_f1']['estimate']:.4f}")
    print(f"Adapted CNN + Feature Vote AUROC: {metrics_A_adapted_fused['auroc']['estimate']:.4f} [{metrics_A_adapted_fused['auroc']['lower_95']:.4f}, {metrics_A_adapted_fused['auroc']['upper_95']:.4f}], F1: {metrics_A_adapted_fused['macro_f1']['estimate']:.4f}")

    # 5. Comparison B: Metadata Gain (Internal to PhysioNet across 5 folds)
    p_oof_2vote = np.zeros(len(merged))
    p_oof_3vote = np.zeros(len(merged))
    w_folds_2vote = []
    w_folds_3vote = []

    for k in range(5):
        train = folds_phys != k
        test = folds_phys == k
        w2 = best_weights(y_phys[train], p_cnn_phys[train], p_lr_phys[train])
        w3 = best_weights(y_phys[train], p_cnn_phys[train], p_lr_phys[train], p_m_phys[train])
        w_folds_2vote.append(w2)
        w_folds_3vote.append(w3)
        p_oof_2vote[test] = w2[0] * p_cnn_phys[test] + w2[1] * p_lr_phys[test]
        p_oof_3vote[test] = w3[0] * p_cnn_phys[test] + w3[1] * p_lr_phys[test] + w3[2] * p_m_phys[test]

    metrics_B_2vote = metric_intervals(y_phys, p_oof_2vote, groups_phys, ["auroc"])
    metrics_B_3vote = metric_intervals(y_phys, p_oof_3vote, groups_phys, ["auroc"])
    gain_B = float(metrics_B_3vote["auroc"]["estimate"] - metrics_B_2vote["auroc"]["estimate"])
    gain_interval = paired_gain_bootstrap(y_phys, p_oof_2vote, p_oof_3vote, groups_phys)

    print("\n=== Comparison B: Metadata Gain ===")
    print(f"CNN + Feature Vote AUROC: {metrics_B_2vote['auroc']['estimate']:.4f} [{metrics_B_2vote['auroc']['lower_95']:.4f}, {metrics_B_2vote['auroc']['upper_95']:.4f}]")
    print(f"CNN + Feature + Metadata Vote AUROC: {metrics_B_3vote['auroc']['estimate']:.4f} [{metrics_B_3vote['auroc']['lower_95']:.4f}, {metrics_B_3vote['auroc']['upper_95']:.4f}]")
    print(f"Metadata AUROC Gain: {gain_B:+.4f} [{gain_interval['lower_95']:+.4f}, {gain_interval['upper_95']:+.4f}] (Target >= +0.02)")

    # Contract verification: ensure sample produces valid FusedOutput
    sample_fused = FusedOutput(
        modality="heart",
        p_abnormal=float(p_oof_3vote[0]),
        votes={
            "cnn": float(p_cnn_phys[0]),
            "features": float(p_lr_phys[0]),
            "metadata": float(p_m_phys[0]),
        },
        weights={
            "cnn": w_folds_3vote[0][0],
            "features": w_folds_3vote[0][1],
            "metadata": w_folds_3vote[0][2],
        },
    )

    out_data = {
        "yaseen_weights": list(w_yaseen),
        "comparison_A": {
            "plain_cnn": metrics_A_plain,
            "fused": metrics_A_fused,
            "adapted_cnn": metrics_A_adapted,
            "adapted_cnn_plus_feature_vote": metrics_A_adapted_fused,
        },
        "comparison_B": {
            "two_vote_weights_per_fold": [list(w) for w in w_folds_2vote],
            "three_vote_weights_per_fold": [list(w) for w in w_folds_3vote],
            "two_vote": metrics_B_2vote,
            "three_vote": metrics_B_3vote,
            "auroc_gain": gain_B,
            "auroc_gain_interval": gain_interval,
        },
    }

    out_path = reports_dir / "heart_fusion.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nSaved updated late fusion report to {out_path}")

    # Task 7: Save reports/heart_fusion_params.json
    full_meta_model, meta_cols = run_metadata_vote()
    save_fusion_params(
        T=T,
        yaseen_weights=list(w_yaseen),
        physionet_weights=list(w_folds_3vote[0]),
        feat_model=full_feat_model,
        feat_cols=feature_cols,
        meta_model=full_meta_model,
        meta_cols=meta_cols,
    )

    return out_data


def paired_gain_bootstrap(y_true, p1, p2, patient_ids, n_resamples=1000, seed=0, confidence=0.95) -> dict:
    """Bootstrap confidence interval for paired difference in AUROC (p2 - p1)."""
    y = np.asarray(y_true, dtype=int)
    ids = np.asarray(patient_ids)
    groups = {}
    for i, patient in enumerate(ids.tolist()):
        groups.setdefault(patient, []).append(i)
    clusters = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        y_samp = y[indices]
        if len(np.unique(y_samp)) < 2:
            continue
        auc1 = float(classification_metrics(y_samp, p1[indices])["auroc"])
        auc2 = float(classification_metrics(y_samp, p2[indices])["auroc"])
        gain = auc2 - auc1
        if np.isfinite(gain):
            values.append(gain)
    alpha = (1 - confidence) / 2
    bounds = np.quantile(values, [alpha, 1 - alpha]) if values else [np.nan, np.nan]
    orig_gain = float(classification_metrics(y, p2)["auroc"] - classification_metrics(y, p1)["auroc"])
    return {
        "estimate": orig_gain,
        "lower_95": float(bounds[0]),
        "upper_95": float(bounds[1]),
        "valid_resamples": len(values),
        "resamples": n_resamples,
    }


def save_fusion_params(
    T: float,
    yaseen_weights: list[float],
    physionet_weights: list[float],
    feat_model,
    feat_cols: list[str],
    meta_model,
    meta_cols: list[str],
    out_path: Path = Path("reports/heart_fusion_params.json"),
) -> dict:
    """Task 7: Save fitted parameters as plain numbers into reports/heart_fusion_params.json."""
    imputer_f = feat_model.named_steps["simpleimputer"]
    scaler_f = feat_model.named_steps["standardscaler"]
    lr_f = feat_model.named_steps["logisticregression"]

    imputer_m = meta_model.named_steps["simpleimputer"]
    scaler_m = meta_model.named_steps["standardscaler"]
    lr_m = meta_model.named_steps["logisticregression"]

    params = {
        "T": float(T),
        "final_T": float(T),
        "yaseen_weights": [float(w) for w in yaseen_weights],
        "physionet_weights": [float(w) for w in physionet_weights],
        "feature_vote": {
            "columns": list(feat_cols),
            "medians": [float(x) for x in imputer_f.statistics_],
            "scaler_mean": [float(x) for x in scaler_f.mean_],
            "scaler_scale": [float(x) for x in scaler_f.scale_],
            "coefficients": [float(x) for x in lr_f.coef_[0]],
            "intercept": float(lr_f.intercept_[0]),
        },
        "metadata_vote": {
            "columns": list(meta_cols),
            "medians": [float(x) for x in imputer_m.statistics_],
            "scaler_mean": [float(x) for x in scaler_m.mean_],
            "scaler_scale": [float(x) for x in scaler_m.scale_],
            "coefficients": [float(x) for x in lr_m.coef_[0]],
            "intercept": float(lr_m.intercept_[0]),
        },
    }
    out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Saved fusion parameters to {out_path}")
    return params


@lru_cache(maxsize=1)
def load_fusion_params() -> dict:
    """Load fitted numbers from reports/heart_fusion_params.json."""
    params_path = Path("reports/heart_fusion_params.json")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}. Run cardio.fusion.late_fusion to generate it.")
    return json.loads(params_path.read_text(encoding="utf-8"))


def _predict_vote(x_dict: dict, vote_params: dict) -> float:
    """Compute logistic regression probability from a feature dictionary using fitted parameters."""
    cols = vote_params["columns"]
    medians = vote_params["medians"]
    means = vote_params["scaler_mean"]
    scales = vote_params["scaler_scale"]
    coefs = vote_params["coefficients"]
    intercept = vote_params["intercept"]

    z = float(intercept)
    for i, col in enumerate(cols):
        val = x_dict.get(col)
        if val is None or not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = medians[i]
        scaled = (float(val) - means[i]) / scales[i]
        z += coefs[i] * scaled
    p = 1.0 / (1.0 + math.exp(-max(min(z, 500.0), -500.0)))
    return float(max(min(p, 1.0), 0.0))


def heart_fusion(
    p_cnn: float,
    features: Optional[dict] = None,
    metadata: Optional[PatientMetadata | dict] = None,
) -> FusedOutput:
    """Task 7: Combines CNN probability, feature vote, and metadata vote into FusedOutput.

    Uses fitted numbers saved in reports/heart_fusion_params.json.
    Missing votes are omitted and remaining weights are normalised to sum to 1.
    """
    params = load_fusion_params()
    votes: dict[str, float] = {"cnn": float(max(min(p_cnn, 1.0), 0.0))}

    # 1. Feature vote
    if features is not None and isinstance(features, dict) and len(features) > 0:
        feat_keys = set(params["feature_vote"]["columns"])
        if any(k in feat_keys and features[k] is not None for k in features):
            p_feat = _predict_vote(features, params["feature_vote"])
            votes["features"] = p_feat

    # 2. Metadata vote
    if metadata is not None:
        if isinstance(metadata, dict):
            raw_age = metadata.get("age_years", metadata.get("age"))
            raw_sex = metadata.get("sex")
            raw_h = metadata.get("height_cm", metadata.get("height"))
            raw_w = metadata.get("weight_kg", metadata.get("weight"))
            raw_p = metadata.get("pregnant", metadata.get("pregnancy"))
        else:
            raw_age = getattr(metadata, "age_years", None)
            raw_sex = getattr(metadata, "sex", None)
            raw_h = getattr(metadata, "height_cm", None)
            raw_w = getattr(metadata, "weight_kg", None)
            raw_p = getattr(metadata, "pregnant", None)

        meta_dict = {}
        if raw_age is not None:
            if isinstance(raw_age, (int, float)) and math.isfinite(raw_age):
                if raw_age < 0.08:
                    meta_dict["age"] = 0.0
                elif raw_age <= 1.0:
                    meta_dict["age"] = 1.0
                elif raw_age <= 12.0:
                    meta_dict["age"] = 2.0
                elif raw_age <= 18.0:
                    meta_dict["age"] = 3.0
                else:
                    meta_dict["age"] = 4.0
            else:
                mapped = AGE_MAP.get(str(raw_age))
                if mapped is not None:
                    meta_dict["age"] = float(mapped)

        if raw_sex is not None:
            if raw_sex in ("female", "Female", 0, "0"):
                meta_dict["sex"] = 0.0
            elif raw_sex in ("male", "Male", 1, "1"):
                meta_dict["sex"] = 1.0

        if raw_h is not None and math.isfinite(float(raw_h)):
            meta_dict["height"] = float(raw_h)

        if raw_w is not None and math.isfinite(float(raw_w)):
            meta_dict["weight"] = float(raw_w)

        if raw_p is not None:
            if raw_p is True or raw_p in (1, "True", "true"):
                meta_dict["pregnant"] = 1.0
            elif raw_p is False or raw_p in (0, "False", "false"):
                meta_dict["pregnant"] = 0.0

        if len(meta_dict) > 0:
            p_meta = _predict_vote(meta_dict, params["metadata_vote"])
            votes["metadata"] = p_meta

    # 3. Choose weights and normalise
    has_feat = "features" in votes
    has_meta = "metadata" in votes

    phys_w = params.get("physionet_weights", [0.70, 0.0, 0.30])
    yaseen_w = params.get("yaseen_weights", [1.0, 0.0, 0.0])

    if has_feat and has_meta:
        raw_w = {"cnn": float(phys_w[0]), "features": float(phys_w[1]), "metadata": float(phys_w[2])}
    elif has_feat and not has_meta:
        raw_w = {"cnn": float(yaseen_w[0]), "features": float(yaseen_w[1])}
    elif not has_feat and has_meta:
        raw_w = {"cnn": float(phys_w[0]), "metadata": float(phys_w[2])}
    else:
        raw_w = {"cnn": 1.0}

    total_w = sum(raw_w.values())
    weights = {k: float(w / total_w) for k, w in raw_w.items()}

    p_abnormal = float(sum(votes[k] * weights[k] for k in votes))
    p_abnormal = float(max(min(p_abnormal, 1.0), 0.0))

    return FusedOutput(
        modality="heart",
        p_abnormal=p_abnormal,
        votes=votes,
        weights=weights,
    )


def main():
    run_late_fusion()


if __name__ == "__main__":
    main()

