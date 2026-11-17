# Harsh: S1/S2/S4/S5 rework (17 Nov)

Bugs found in the earlier version and fixed:

- **CLAHE numbers were wrong.** `features.py` computed three standard deviations at clip limits 1, 2 and 4 and rounded
  them to 1 decimal. The roadmap asks for mean, standard deviation and mean absolute Laplacian of the lung area at clip 2.0,
  8x8 tiles, from `lung/model/clahe.py` (which did not exist). Columns are now `clahe_mean`, `clahe_std`,
  `clahe_abs_laplacian`; one function (`lung.model.features.characteristics`) feeds the CSV, `common.image_features`
  and `lung/predict.py`.
- **No balanced class weights** in the feature vote, the metadata vote and the RSNA comparison (roadmap section 2).
- **ECE used 10 bins** in `calibrate.py` (roadmap: 15) and a different definition from `metrics.py`; now one definition, 15 bins.
- **Fusion used the uncalibrated CNN** and took one temperature for all folds. Now T and the weights for fold k come from
  the other four folds only; fused output keeps per-fold T and weights.
- **Metadata vote** filled `age` and `age^2` with separate means (so `age^2` was not the square of `age`) and treated
  unknown sex as female; it now uses `common.meta_matrix` (median age fill from training folds, unknown sex 0.5).
- **Adapted model never reached the comparison**: nothing wrote `xray_cnn_adapted_rsna.csv`. Added
  `lung/eval/run_adapted_rsna.py`; the images used to adapt are excluded from every generalisation score.
- **`lung/predict.py` fusion was hard-coded** (`p_M = 0.55 if male`, weights 0.9/0.1, T 0.9442). It now calls
  `xray_fusion` with `reports/xray_fusion_params.json`.

New: `lung/model/clahe.py`, `xray_fusion()` and `reports/xray_fusion_params.json`, `lung/run_all.py`,
`tests/test_lung_harsh.py`.

Open problems, not fixable from this laptop:

- **`data/manifests/rsna_manifest.csv` has no ages** (all 14,863 empty). `stage_2_detailed_class_info.csv` carries no age;
  it is in the DICOM headers. So the metadata vote is sex-only (AUROC 0.507) and the S2 metadata-gain target
  (+0.02) cannot be judged until `lung/data/build_rsna.py` is re-run with pydicom on the RSNA files (Vedansh's script).
- RSNA images are not on this laptop, so `run_rsna`, `rsna_adapted`, `features_rsna` and `fusion_rsna` were not re-run.
  `reports/xray_fusion.json` and the `app_weights` in `reports/xray_fusion_params.json` still come from the old
  feature columns; re-run `python -m lung.run_all --from rsna_cnn` on the laptop that has RSNA.

## Update: RSNA run with ages (17 Nov, later)

RSNA images and headers are now on this laptop. Done:

- `lung/data/build_rsna.py` read ages wrongly: the RSNA headers hold a bare number (`'51'`), the parser only accepted
  `051Y`. Vedansh pushed the same fix to main (PR 26, `lung-vedansh-fix-rsna-age`) while this was running, and that
  version, including its rebuilt `data/manifests/rsna_manifest.csv`, is the one used here (14,860 patients, all with an
  age; same patients, labels, ages and sex as my own rebuild). Its folds are a fresh stratified assignment and
  differ from the earlier committed folds for about 41% of patients, so every fold-dependent step was re-run on it;
  the S2 numbers below did not change in any decimal that matters (metadata gain +0.0140 either way). The only change
  kept from my side is that the script also finds the files in `data/raw/rsna/rsna-pneumonia-detection-challenge/`.
  `lung/predict.py` takes T from `reports/xray_fusion_params.json` (falls back to the calibration report).
- `run_fusion_rsna.py` used the uncalibrated CNN and its own copy of the metadata vote; it now calibrates with T and uses
  `lung.fusion.meta_vote.oof_metadata_vote` (one definition).
- `python -m lung.run_all`: all 17 steps have run on the final code. The first run stopped at `adaptation` with exit
  code 4 and no Python error; running `lung.eval.run_tta` alone passed (exit 0), so it was a one-off, and the run was
  resumed from `robustness`. For a strictly single pass, run `python -m lung.run_all` once more after the last change.

Results against the S2/S4 targets (all with 95% bootstrap intervals in `reports/xray_fusion.json`):

| Target | Result | Met |
|---|---|---|
| ECE after temperature scaling <= 0.05 | 0.0269 -> 0.0247 (T = 0.944) | yes |
| Metadata gain >= +0.02 AUROC, w3 <= 0.3 | +0.0140 [0.0119, 0.0163], mean w3 0.15; vote alone AUROC 0.536 | **no** |
| Generalisation >= +0.03 AUROC | CNN 0.9070 -> CNN + features 0.8986 (-0.0084), feature weight 0.05 | **no** |
| Adaptation recovers >= 50% (N <= 64) | RSNA: hurts at every N (AUROC 0.907 plain; 0.873, 0.829, 0.867, 0.841, 0.839 adapted for N = 8..128); synthetic Kermany shift: 94.5% recovered | **no on RSNA** |

What did not work, and why (not tuned away; no RSNA label was used to choose anything above):

- Age and sex say very little about lung opacity in RSNA, so the metadata vote's gain stays below +0.02 and the
  interval does not reach it.
- The feature vote is learned on children (Kermany) and applied to adults; Kermany-chosen weights give it 0.05 and it
  slightly lowers RSNA AUROC.
- BN adaptation fixes a brightness/contrast shift (synthetic shift) but makes RSNA worse for all N and all 5 draws, so
  the "full branch" (adapted CNN + features, AUROC 0.8456 on the scored images) is below the plain CNN. Larger N does
  not help. The dashboard should keep the plain 8-bit model, not an adapted one.
- `reports/xray_fusion_params.json` now has weights cnn 0.85 / features 0.00 / metadata 0.15 (chosen on RSNA
  out-of-fold votes; a demo setting, not an evaluation).

Not mine, but found: the adapted 8-bit model agrees with the float model on only 95.69% of fold 0 (target 98%;
`reports/xray_edge_budget.json`); the original is at 99.01%. Needs Mihir's re-quantisation with more representative images.

## Adapted model in the comparison (S4, Harsh)

`xray_fusion.json` -> `generalisation` already held adapted CNN and adapted CNN + features (the full branch). Added
`metadata_gain.with_adapted_cnn`, so every version exists with and without adaptation (roadmap section 5): adapted
CNN + features + metadata AUROC 0.8398 (plain version 0.9076); metadata gain with the adapted CNN -0.0028, i.e. none.
Full branch against the plain CNN: 0.8456 vs 0.9070 (-0.0614 [-0.0667, -0.0561]). One adaptation draw (N = 32, seed 0,
32 images left out of every score); the five-draw picture for N = 8..128 is in `reports/xray_tta.json`. The full branch
is worse than the plain CNN on RSNA, so the +0.03 generalisation target is missed with or without adaptation.
