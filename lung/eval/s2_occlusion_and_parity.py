from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from core.descriptors import vision_zone_attribution
from core.images import features_for_image
from core.quantize import tflite_predict

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"


def bootstrap_agreement(preds_a: np.ndarray, preds_b: np.ndarray,
                        patient_ids: np.ndarray, n_boot: int = 1000,
                        seed: int = 0) -> dict:
    item_agree = float(np.mean(preds_a == preds_b))

    df = pd.DataFrame({"pid": patient_ids, "a": preds_a, "b": preds_b})
    pat_df = df.groupby("pid", as_index=False)[["a", "b"]].mean()
    pat_a = (pat_df["a"].values >= 0.5).astype(int)
    pat_b = (pat_df["b"].values >= 0.5).astype(int)
    pat_agree = float(np.mean(pat_a == pat_b))

    rng = np.random.default_rng(seed)
    n_pat = len(pat_df)
    pat_boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_pat, size=n_pat)
        pat_boots[i] = np.mean(pat_a[idx] == pat_b[idx])

    n_items = len(preds_a)
    item_boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        item_boots[i] = np.mean(preds_a[idx] == preds_b[idx])

    return {
        "n_images": int(n_items),
        "n_patients": int(n_pat),
        "item_agreement": round(item_agree, 4),
        "item_agreement_ci95": [
            round(float(np.percentile(item_boots, 2.5)), 4),
            round(float(np.percentile(item_boots, 97.5)), 4),
        ],
        "patient_agreement": round(pat_agree, 4),
        "patient_agreement_ci95": [
            round(float(np.percentile(pat_boots, 2.5)), 4),
            round(float(np.percentile(pat_boots, 97.5)), 4),
        ],
        "discordant_images": int(np.sum(preds_a != preds_b)),
        "discordant_patients": int(np.sum(pat_a != pat_b)),
    }


def ensure_4d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X[..., None]
    if X.ndim == 4 and X.shape[-1] == 1:
        return X
    if X.ndim == 4 and X.shape[1] == 1:
        return np.transpose(X, (0, 2, 3, 1))
    raise ValueError(f"Unexpected image tensor shape: {X.shape}")


def quantize_inputs(X_norm: np.ndarray, in_scale: float, in_zp: int) -> np.ndarray:
    q = np.round(X_norm / in_scale + in_zp)
    return np.clip(q, -128, 127).astype(np.int8)


def compute_occlusion_records(interp: tf.lite.Interpreter, X_int8: np.ndarray,
                              pids: np.ndarray, labels: np.ndarray,
                              q_probs: np.ndarray, dataset_name: str,
                              sample_indices: np.ndarray) -> list[dict]:
    records = []
    for idx in sample_indices:
        res = vision_zone_attribution(interp, X_int8[idx : idx + 1], grid=3)
        records.append({
            "dataset": dataset_name,
            "index": int(idx),
            "patient_id": str(pids[idx]),
            "true_label": int(labels[idx]),
            "int8_predicted_class": int(res["predicted_class"]),
            "int8_p_pneumonia": round(float(q_probs[idx, 1]), 4),
            "base_confidence": float(res["base_confidence"]),
            "peak_region": res["peak_region"],
            "heatmap_json": json.dumps(res["heatmap"]),
        })
    return records


def load_rsna_subset_if_available():
    manifest_path = MANIFEST_DIR / "rsna_manifest.csv"
    if not manifest_path.exists():
        return None

    df = pd.read_csv(manifest_path)
    rsna_96_dir = CACHE_DIR / "rsna_96"

    if rsna_96_dir.exists():
        npy_exists = df["patient_id"].apply(lambda pid: (rsna_96_dir / f"{pid}.npy").exists())
        df_avail = df[npy_exists].reset_index(drop=True)
        if len(df_avail) > 0:
            X = np.stack([np.load(rsna_96_dir / f"{pid}.npy") for pid in df_avail["patient_id"]], axis=0)
            y = df_avail["label"].values.astype(np.int64)
            pids = df_avail["patient_id"].values
            return X, y, pids, f"rsna_96 cache ({len(df_avail)} images)"

    existing_mask = df["path"].apply(lambda p: (REPO_ROOT / p).exists())
    df_avail = df[existing_mask].reset_index(drop=True)
    if len(df_avail) == 0:
        return None

    X = np.stack([features_for_image(str(REPO_ROOT / rel_path)) for rel_path in df_avail["path"]], axis=0)
    y = df_avail["label"].values.astype(np.int64)
    pids = df_avail["patient_id"].values
    return X, y, pids, f"png256 ({len(df_avail)} images)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rsna-occ-samples", type=int, default=100,
                    help="Number of stratified RSNA images for occlusion maps (default: 100)")
    ap.add_argument("--recompute-kermany", action="store_true",
                    help="Force recomputation of Kermany Fold 0 even if already saved")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stem = "chiron_pulmo_v1"
    float_path = MODEL_DIR / f"{stem}_float.keras"
    tfl_path = MODEL_DIR / f"{stem}_int8.tflite"
    quant_rep_path = REPORT_DIR / f"quantization_{stem}.json"
    summary_path = REPORT_DIR / "xray_s2_int8_parity_and_occlusion.json"
    occ_csv_path = REPORT_DIR / "xray_occlusion_8bit.csv"

    for p in (float_path, tfl_path, quant_rep_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")

    qmeta = json.loads(quant_rep_path.read_text())
    mu = float(qmeta["norm_mean"])
    sd = float(qmeta["norm_std"])
    in_scale = float(qmeta["input_scale"])
    in_zp = int(qmeta["input_zero_point"])

    float_model = keras.models.load_model(float_path)
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    if not args.recompute_kermany and summary_path.exists() and occ_csv_path.exists():
        existing_summary = json.loads(summary_path.read_text())
        existing_occ_df = pd.read_csv(occ_csv_path)
        k_df = existing_occ_df[existing_occ_df["dataset"] == "kermany_fold0"]
        if existing_summary.get("kermany_fold0_agreement") and len(k_df) > 0:
            k_agree = existing_summary["kermany_fold0_agreement"]
            k_region_summary = existing_summary["kermany_fold0_occlusion_peak_regions"]
            k_occ_records = k_df.to_dict(orient="records")
        else:
            args.recompute_kermany = True
    else:
        args.recompute_kermany = True

    if args.recompute_kermany:
        k_cache = np.load(CACHE_DIR / "kermany_logmel.npz", allow_pickle=True)
        te = k_cache["fold"] == 0
        X_k = ensure_4d((k_cache["X"][te] - mu) / sd).astype(np.float32)
        y_k = k_cache["y"][te]
        pid_k = k_cache["patient_id"][te]

        f_prob_k = float_model.predict(X_k, verbose=0).reshape(len(X_k), -1)
        q_prob_k = tflite_predict(interp, X_k)

        k_agree = bootstrap_agreement(
            f_prob_k.argmax(1), q_prob_k.argmax(1), pid_k, n_boot=1000, seed=args.seed
        )
        print("Kermany Fold 0: Float32 vs INT8 Agreement")
        print(json.dumps(k_agree, indent=2))

        print(f"Computing 8-bit vision_zone_attribution for Kermany Fold 0 images...")
        X_k_int8 = quantize_inputs(X_k, in_scale, in_zp)
        k_occ_records = compute_occlusion_records(
            interp, X_k_int8, pid_k, y_k, q_prob_k,
            dataset_name="kermany_fold0",
            sample_indices=np.arange(len(X_k)),
        )
        k_occ_df = pd.DataFrame(k_occ_records)
        k_region_summary = (
            k_occ_df.groupby(["int8_predicted_class", "peak_region"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
    rsna_data = load_rsna_subset_if_available()
    rsna_agree = None
    rsna_occ_records = []
    rsna_region_summary = None

    if rsna_data is None:
        print("files not found")
    else:
        X_r_raw, y_r, pid_r, src_desc = rsna_data
        print(f"RSNA: Float32 vs INT8 Agreement")
        X_r = ensure_4d((X_r_raw - mu) / sd).astype(np.float32)
        f_prob_r = float_model.predict(X_r, batch_size=64, verbose=0).reshape(len(X_r), -1)
        q_prob_r = tflite_predict(interp, X_r)

        rsna_agree = bootstrap_agreement(
            f_prob_r.argmax(1), q_prob_r.argmax(1), pid_r, n_boot=1000, seed=args.seed
        )
        print(json.dumps(rsna_agree, indent=2))

        rng = np.random.default_rng(args.seed)
        half = args.rsna_occ_samples // 2
        idx0 = rng.choice(np.flatnonzero(y_r == 0), size=min(half, int((y_r == 0).sum())), replace=False)
        idx1 = rng.choice(np.flatnonzero(y_r == 1), size=min(half, int((y_r == 1).sum())), replace=False)
        sel_r = np.sort(np.concatenate([idx0, idx1]))

        print("Computing 8-bit vision_zone_attribution for stratified RSNA images...")
        X_r_int8 = quantize_inputs(X_r[sel_r], in_scale, in_zp)
        rsna_occ_records = compute_occlusion_records(
            interp, X_r_int8, pid_r[sel_r], y_r[sel_r], q_prob_r[sel_r],
            dataset_name="rsna_sample",
            sample_indices=np.arange(len(sel_r)),
        )
        r_occ_df = pd.DataFrame(rsna_occ_records)
        rsna_region_summary = (
            r_occ_df.groupby(["int8_predicted_class", "peak_region"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

    REPORT_DIR.mkdir(exist_ok=True)
    summary_report = {
        "model": stem,
        "seed": args.seed,
        "kermany_fold0_agreement": k_agree,
        "kermany_fold0_occlusion_peak_regions": k_region_summary,
        "rsna_agreement": rsna_agree,
        "rsna_occlusion_peak_regions": rsna_region_summary,
    }
    summary_path.write_text(json.dumps(summary_report, indent=2))

    all_occ = k_occ_records + rsna_occ_records
    pd.DataFrame(all_occ).to_csv(occ_csv_path, index=False)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {occ_csv_path} ({len(all_occ)} occlusion maps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
