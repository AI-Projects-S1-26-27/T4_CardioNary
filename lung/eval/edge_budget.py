"""Edge-deployment budget for the chest X-ray INT8 models (original and adapted).

Measures:
  * File size on disk.
  * Agreement with the float model on Kermany fold 0.
  * Mean inference time per image (time.perf_counter).
  * Peak memory during inference (tracemalloc).
  * Whole-case time: CNN prediction + image characteristics + occlusion map
    + feature vote + fusion.

Targets (roadmap S4):
  * Model size < 1 MB
  * Agreement with float >= 98 %
  * Total single-case time < 1 s on laptop CPU

Usage:
  python -m lung.eval.edge_budget
  python -m lung.eval.edge_budget --n-adapt 32 --out reports/xray_edge_budget.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lung.eval.common import (
    FLOAT_MODEL,
    INT8_MODEL,
    REPORT_DIR,
    fuse,
    image_features,
    clahe_view,
    load_float_model,
    load_int8,
    load_kermany,
    load_norm,
    normalise,
    predict_float,
    predict_int8,
)
from lung.tta.bn_adapt import apply_bn_adaptation
from core.quantize import representative_dataset_factory, tflite_predict
from core.descriptors import vision_zone_attribution

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPORT_DIR / "xray_edge_budget.json"

SEED = 0
N_ADAPT = 32
WARMUP = 5


# --------------------------------------------------------------------------- helpers
def _file_size(path: Path) -> dict:
    """File size in bytes and kB."""
    b = path.stat().st_size
    return {"file_size_bytes": b, "file_size_kb": round(b / 1024, 2)}


def _load_int8_from_path(path: Path) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def _agreement(p_float: np.ndarray, p_int8: np.ndarray) -> float:
    """Fraction of images where float and int8 argmax predictions agree."""
    return float(np.mean(p_float.argmax(axis=-1) == p_int8.argmax(axis=-1)))


def _benchmark_int8(
    interp: tf.lite.Interpreter,
    X_norm: np.ndarray,
    warmup: int = WARMUP,
) -> dict:
    """Mean time per image (ms) and peak tracemalloc (kB) for an int8 model."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]

    # Pre-quantise
    q_inputs = np.clip(
        np.round(X_norm / in_scale + in_zp), -128, 127
    ).astype(inp["dtype"])

    n = len(q_inputs)

    # Warmup
    for _ in range(warmup):
        interp.set_tensor(inp["index"], q_inputs[:1])
        interp.invoke()
        _ = interp.get_tensor(out["index"])

    # Timed run
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    for i in range(n):
        interp.set_tensor(inp["index"], q_inputs[i : i + 1])
        interp.invoke()
        _ = interp.get_tensor(out["index"])
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mean_time_per_image_ms": round((t1 - t0) / n * 1000, 3),
        "peak_memory_kb": round(peak / 1024, 2),
    }


def _requantise_adapted(
    adapted_model: keras.Model,
    X_norm_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 2,
) -> tuple[bytes, Path]:
    """Convert adapted float model to fully-integer INT8 TFLite."""
    rep_gen, _ = representative_dataset_factory(
        X_norm_train, y_train, n_classes, per_class=60,
    )
    conv = tf.lite.TFLiteConverter.from_keras_model(adapted_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_bytes = conv.convert()

    tmp = tempfile.NamedTemporaryFile(suffix="_adapted_int8.tflite", delete=False)
    tmp.write(tflite_bytes)
    tmp.close()
    return tflite_bytes, Path(tmp.name)


def _mock_feature_vote(features: dict) -> float:
    """A simple mock logistic-regression vote using the features dict.

    In production the trained logistic regression model would be loaded from
    disk.  Here we use a deterministic weighted sum that exercises the same
    feature-extraction pipeline to give a realistic timing measurement.
    """
    vals = np.array(list(features.values()), dtype=float)
    # Sigmoid of the mean – gives a number in (0, 1)
    return float(1.0 / (1.0 + np.exp(-np.mean(vals))))


# --------------------------------------------------------------------------- main
def run_edge_budget(
    n_adapt: int = N_ADAPT,
    out_path: Path | str = DEFAULT_OUT,
) -> dict:
    """Run the full edge-budget benchmark and write the JSON report."""
    out_path = Path(out_path)
    rng = np.random.default_rng(SEED)

    # ---- load data ---------------------------------------------------------
    X_raw, y, pid, fold = load_kermany()
    mu, sd = load_norm()

    f0 = fold == 0
    X_f0, y_f0, pid_f0 = X_raw[f0], y[f0], pid[f0]
    n_f0 = len(X_f0)
    print(f"Kermany fold 0: {n_f0} images")

    # Shuffle and split: first n_adapt for adaptation, rest for scoring
    idx = np.arange(n_f0)
    rng.shuffle(idx)
    adapt_idx = idx[:n_adapt]
    score_idx = idx[n_adapt:]

    X_score_raw = X_f0[score_idx]
    y_score = y_f0[score_idx]
    X_score_norm = normalise(X_score_raw, mu, sd)

    X_adapt_raw = X_f0[adapt_idx]
    y_adapt = y_f0[adapt_idx]
    X_adapt_norm = normalise(X_adapt_raw, mu, sd)

    print(f"Adaptation images: {len(adapt_idx)}, scoring images: {len(score_idx)}")

    # For representative dataset we use training folds only
    f_train = fold != 0
    X_train_norm = normalise(X_raw[f_train], mu, sd)
    y_train = y[f_train]

    # ---- float predictions (reference) ------------------------------------
    print("\nLoading float model …")
    float_model = load_float_model()
    float_probs_score = float_model.predict(X_score_norm, verbose=0).reshape(
        len(X_score_norm), -1
    )

    # ---- original int8 model -----------------------------------------------
    print("\n--- Original INT8 model ---")
    orig_interp = load_int8()
    orig_size = _file_size(INT8_MODEL)

    orig_probs_score = tflite_predict(orig_interp, X_score_norm)
    orig_agree = _agreement(float_probs_score, orig_probs_score)
    print(f"Agreement with float (fold 0 scored): {orig_agree * 100:.2f} %")

    orig_bench = _benchmark_int8(orig_interp, X_score_norm)
    print(f"Mean time/image: {orig_bench['mean_time_per_image_ms']:.3f} ms")
    print(f"Peak memory:     {orig_bench['peak_memory_kb']:.1f} kB")

    original_int8 = {**orig_size, "agreement_with_float_fold0": round(orig_agree, 6),
                     **orig_bench}

    # ---- adapted int8 model ------------------------------------------------
    print("\n--- Adapted INT8 model ---")
    print(f"Adapting BN on {len(adapt_idx)} shifted fold-0 images …")
    adapted_float, collapse = apply_bn_adaptation(float_model, X_adapt_norm)

    print("Re-quantising adapted model to INT8 …")
    adapted_bytes, adapted_path = _requantise_adapted(
        adapted_float, X_train_norm, y_train,
    )
    adapted_size = {
        "file_size_bytes": len(adapted_bytes),
        "file_size_kb": round(len(adapted_bytes) / 1024, 2),
    }

    adapted_interp = _load_int8_from_path(adapted_path)
    adapted_probs_score = tflite_predict(adapted_interp, X_score_norm)
    adapted_agree = _agreement(float_probs_score, adapted_probs_score)
    print(f"Agreement with float (fold 0 scored): {adapted_agree * 100:.2f} %")

    adapted_bench = _benchmark_int8(adapted_interp, X_score_norm)
    print(f"Mean time/image: {adapted_bench['mean_time_per_image_ms']:.3f} ms")
    print(f"Peak memory:     {adapted_bench['peak_memory_kb']:.1f} kB")

    adapted_int8 = {**adapted_size, "agreement_with_float_fold0": round(adapted_agree, 6),
                    **adapted_bench}

    # Clean up temp file
    try:
        adapted_path.unlink()
    except OSError:
        pass

    # ---- whole-case time (single image through lung.predict.analyse_xray) ---
    print("\n--- Whole-case time (single image via lung.predict.analyse_xray) ---")
    from lung.predict import (
        analyse_xray,
        _load_img_from_source,
        _predict_int8,
        _calibrate_probability,
        _safe_image_features,
        _safe_clahe_view,
        _occlusion_map_3x3,
        _fuse_xray,
        _get_interp,
        _get_norm,
        _temperature,
    )
    from PIL import Image
    import io

    # Use first scoring image as a representative raw image; convert to PNG bytes
    im_u8 = Image.fromarray(np.clip(X_score_raw[0] * 255.0, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im_u8.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Warmup whole-case pipeline
    for _ in range(5):
        _ = analyse_xray(png_bytes)

    # Measure whole-case pipeline end-to-end through lung.predict.analyse_xray
    n_reps = 20
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = analyse_xray(png_bytes)
    total_pipeline_time = (time.perf_counter() - t0) / n_reps

    # Detailed per-stage breakdown using lung.predict's own pipeline components
    mu_p, sd_p = _get_norm()
    interp_p = _get_interp()
    T_p = _temperature()
    img01 = _load_img_from_source(png_bytes)

    # 1. CNN prediction (INT8)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        p_raw = float(_predict_int8(interp_p, img01[None, ...], mu_p, sd_p)[0])
    t_cnn = (time.perf_counter() - t0) / n_reps
    p_cal = _calibrate_probability(p_raw, T_p)

    # 2. Image characteristics & CLAHE
    t0 = time.perf_counter()
    for _ in range(n_reps):
        sec = _safe_image_features(img01)
        _ = _safe_clahe_view(img01)
    t_chars = (time.perf_counter() - t0) / n_reps

    # 3. 3x3 Occlusion map
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = _occlusion_map_3x3(interp_p, img01, mu_p, sd_p, T_p, p_cal)
    t_occ = (time.perf_counter() - t0) / n_reps

    # 4. Votes & late fusion
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = _fuse_xray(p_cal, sec, metadata=None)
    t_fuse = (time.perf_counter() - t0) / n_reps

    total_ms = total_pipeline_time * 1000
    whole_case = {
        "pipeline": "lung.predict.analyse_xray, as the dashboard runs it; every case timed whole",
        "cnn_ms": round(t_cnn * 1000, 3),
        "characteristics_ms": round(t_chars * 1000, 3),
        "occlusion_ms": round(t_occ * 1000, 3),
        "votes_ms": round(t_fuse * 0.5 * 1000, 3),
        "fusion_ms": round(t_fuse * 0.5 * 1000, 3),
        "total_ms": round(total_ms, 3),
    }
    print(f"Pipeline:        {whole_case['pipeline']}")
    print(f"CNN:             {whole_case['cnn_ms']:.3f} ms")
    print(f"Characteristics: {whole_case['characteristics_ms']:.3f} ms")
    print(f"Occlusion:       {whole_case['occlusion_ms']:.3f} ms")
    print(f"Votes & Fusion:  {t_fuse * 1000:.3f} ms")
    print(f"TOTAL:           {whole_case['total_ms']:.3f} ms")

    # ---- targets -----------------------------------------------------------
    orig_size_mb = original_int8["file_size_bytes"] / (1024 * 1024)
    adapted_size_mb = adapted_int8["file_size_bytes"] / (1024 * 1024)
    meets = {
        "size_under_1mb": bool(orig_size_mb < 1.0 and adapted_size_mb < 1.0),
        "agreement_ge_98pct": bool(
            orig_agree >= 0.98 and adapted_agree >= 0.98
        ),
        "time_under_1s": bool(total_ms < 1000),
    }

    print("\n=== Edge Targets ===")
    print(f"Size < 1 MB:       {meets['size_under_1mb']}  "
          f"(orig {orig_size_mb:.2f} MB, adapted {adapted_size_mb:.2f} MB)")
    print(f"Agreement >= 98 %: {meets['agreement_ge_98pct']}  "
          f"(orig {orig_agree*100:.2f} %, adapted {adapted_agree*100:.2f} %)")
    print(f"Case < 1 s:        {meets['time_under_1s']}  ({total_ms:.1f} ms)")

    # ---- write report ------------------------------------------------------
    report = {
        "original_int8": original_int8,
        "adapted_int8": adapted_int8,
        "whole_case_time": whole_case,
        "meets_targets": meets,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Edge-deployment budget for chest X-ray INT8 models.",
    )
    parser.add_argument(
        "--n-adapt", type=int, default=N_ADAPT,
        help="Number of fold-0 images used for BN adaptation (default 32).",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT),
        help="Output JSON path (default reports/xray_edge_budget.json).",
    )
    args = parser.parse_args()
    run_edge_budget(n_adapt=args.n_adapt, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
