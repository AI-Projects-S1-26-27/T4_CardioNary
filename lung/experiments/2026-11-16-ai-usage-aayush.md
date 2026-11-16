# AI-Usage Note: Chest X-ray Adaptation & Edge Budget Deliverables

**Team Member:** Aayush Pratap Asnani  
**Roles & Tasks:** S1 Adaptation setup & Timing; S3 Test-Time Adaptation; S4 Edge Budget Benchmark; S5 Number Verification; S6 Report Sections (4.3 & Timing)  
**Date:** 16 November 2026  
**Project:** CardioNary (CS F407 Artificial Intelligence)

---

## 1. Tools and Models Used

- **Agent Assistants & LLMs:**
  - Google DeepMind Antigravity AI Assistant (Gemini 3.8 Flash & Claude Opus 4.6)
- **Local Environment & Compute Libraries:**
  - Python 3.11 / 3.10 virtual environment (`.venv`)
  - TensorFlow 2.19.1 / Keras 3.12.4
  - TensorFlow Lite Runtime (`tf.lite.Interpreter`)
  - NumPy 2.1.3, SciPy 1.15.3, pandas 2.3.3
  - scikit-learn 1.7.2 (`roc_auc_score`, `accuracy_score`, `roc_curve`)
  - OpenCV (`opencv-python-headless`), Pillow 12.3.0
  - Python standard library profiling: `time.perf_counter`, `tracemalloc`
  - Testing: `pytest`

---

## 2. What AI Was Used For

1. **Test-Time Adaptation Pipeline Design & Verification (`lung/tta/bn_adapt.py`, `lung/eval/run_tta.py`):**
   - Implemented the 5-step batch-normalisation re-estimation protocol without gradient computation.
   - Designed and verified the collapse guard mechanism (reverting BN statistics if predictions collapse to $> 95\%$ in any single class).
   - Formulated the recovery metric calculation:
     $$\text{Recovery} = \frac{\text{Adapted} - \text{Plain}}{\text{Clean} - \text{Plain}}$$
   - Structured patient-level exclusion masks ensuring no patient in the adaptation batch was ever scored.

2. **Edge Budget & Benchmarking Script Development (`lung/eval/timing.py`, `lung/eval/edge_budget.py`):**
   - Profiled single-case end-to-end execution across all 5 pipeline stages: CNN forward pass, image descriptors, occlusion attribution grid, logistic regression vote, and late fusion.
   - Built the INT8 re-quantisation calibration pipeline for adapted float models using representative datasets.

3. **Scientific Documentation & Audit:**
   - Summarised experiment metrics into markdown tables.
   - Cross-checked committed JSON values against roadmap targets.

---

## 3. How the Output Was Checked and Validated

Every piece of generated code and every empirical metric was strictly verified against ground truth:

1. **Independent Verification on Frozen Datasets:**
   - All benchmark numbers were executed on Kermany 2017 fold 0 and verified with real forward passes through the committed float Keras (`chiron_pulmo_v1_float.keras`) and INT8 TFLite (`chiron_pulmo_v1_int8.tflite`) binaries.
   - No synthetic or fabricated results were accepted; all numbers originated directly from committed JSON reports.

2. **Bootstrap Confidence Interval Validation:**
   - Every AUROC and paired difference was tested with 1,000 percentile bootstrap resamples (`seed=0`).
   - Intervals were verified to be numerically consistent and bounds properly checked.

3. **Leakage & Fairness Checks:**
   - Checked that normalization parameters ($\mu, \sigma$) were frozen from training folds only.
   - Checked that the $N$ adaptation images were excluded by patient ID from the scoring partition.
   - Confirmed that representative calibration datasets for quantisation never touched held-out test data.

4. **Edge Target Cross-Verification:**
   - Verified binary file sizes directly via `stat().st_size`.
   - Latencies were measured with high-resolution monotonic clocks (`time.perf_counter`) after warm-up runs.
   - Peak memory usage was validated using `tracemalloc` to reflect genuine per-inference memory overhead.
