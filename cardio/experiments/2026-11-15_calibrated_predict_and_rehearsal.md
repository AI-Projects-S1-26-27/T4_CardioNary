# Calibrated heart result, and a rehearsal of the final run

Sun 15 Nov, Aviral, roadmap S5. Code: `cardio/predict.py`, `cardio/run_all.py`.
Results: `reports/heart_edge_budget.json`.

- **Calibration in the dashboard's result.** `cardio/predict.py` now applies
  Anshuman's temperature, T = 0.504, which `cardio/fusion/calibrate.py` writes to
  `reports/heart_calibration.json` as `"final_T"`. On the public example
  `33151_AV.wav` the top class stays MVP and p_h rises from 0.78 to 0.81: a
  temperature below 1 sharpens the probabilities without changing the winner.
- **Fusion is still the CNN vote alone.** `reports/heart_fusion_params.json` and
  `heart_fusion()` (roadmap S5, Anshuman) are not in the repository yet. The
  fitted weights in `reports/heart_fusion.json` give the feature vote a weight of
  0, and the metadata vote 0.3 when patient details are present.
- **`cardio/run_all.py`** now lists the inputs Anshuman's late-fusion step reads,
  and what it writes (`reports/heart_fusion.json`, `reports/heart_cnn_physionet.csv`).
- **Rehearsal for the Mon 16 Nov final run.** Features (Yaseen and PhysioNet),
  votes, calibration and late fusion ran in order through `run_all.py` in
  5 minutes on this laptop. `reports/heart_calibration.json`,
  `reports/heart_fusion.json` and `cardio/data/circor_folds.csv` came out
  identical. The four CSVs of characteristics and votes differ from the committed
  ones only by floating-point rounding (largest difference 5 × 10⁻¹⁴, no missing
  value moved), so the committed files were kept.
- **Edge budget again, with T applied.** Median time per case 10.5 ms on Yaseen
  fold 0 and 80.2 ms on the PhysioNet sample (95th percentile 149 ms); all
  three targets are still met.
