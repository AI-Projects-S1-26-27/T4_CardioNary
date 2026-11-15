# S1-S4 result delivery audit

This audit starts from `cardio` at `41bfa29`, after S4 PR #14 was merged.
The purpose is to distinguish missing deliverables from unsuccessful model
results. The roadmap's evaluation protocol and targets remain unchanged.

## Delivery gaps

- S2's BMD-HS characteristics had been generated under `scratch/`, but had
  not been committed. `reports/heart_features_bmdhs.csv` now contains one
  row per public recording, with the manifest path and patient ID. A fresh
  extraction reproduced the earlier local file byte for byte. The committed
  PhysioNet characteristics also match the earlier local extraction.
- S2 and S3 had committed `*_reference.json` experiment records, but lacked
  the required official `heart_unseen.json` and `heart_tta.json` outputs.
  The published team float checkpoint and its normalisation report match the
  earlier reference inputs. Both official outputs are regenerated with those
  published files, rather than treating a filename change as a new experiment.
- `cardio.run_all` did not regenerate BMD-HS characteristics. It now includes
  `features_bmdhs` before the fusion step.
- S4's official `heart_robustness.json` and code are already on `cardio`
  through PR #14. They were not still awaiting that merge at audit time.
- Previously passing tests did not require these official deliverables to
  exist. New tests require them, check their checkpoint hashes, join feature
  rows to manifests, and validate every required sweep condition and interval.

Reproduce the public-data runs from the repository root:

```powershell
python -m cardio.run_all --only features_bmdhs features_physionet unseen adaptation robustness
python -m pytest tests/
```

Raw datasets, caches, private clinic data and adapted weights stay local.
The committed feature CSV contains derived characteristics of the public
BMD-HS recordings, not audio or log-mel caches.

## Verification record

The fresh S2 and complete S3 reports reproduce every earlier score, cohort,
confidence interval and guard outcome, excluding the checkpoint-status label
and newly measured timing. All 140 S3 draws were checked for disjoint scoring
patients and the requested stress-batch composition. The longest measured
CPU adaptation including its guard was 1.095 s in this run; the per-draw timings
are retained in `heart_tta.json`.

A fresh S4 replay matches `heart_robustness.json` exactly, including all 75 draw
counts and all intervals. Its five adaptation batches are separate from the
scoring clips. Six public recordings (three per unseen dataset, selected with
seed 0) also reproduce their cached front-end arrays exactly when re-extracted
from the WAV files. The complete test suite additionally checks every local
unseen cache's shape, dtype, finiteness and manifest path.

Final local validation: `python -m pytest tests/ -q -ra --tb=short` passed
158 tests and 7 subtests, with 4 skips (three unavailable lung caches and one
inapplicable image-windowing test). The official-report checks run without raw
datasets, so missing result files can no longer silently pass on a fresh clone.

## Evaluation checks and limits

S1 uses the shared `CARDIO_CONFIG` front end. Metric tests cover known arrays,
ties, missing classes, patient-cluster resampling and exact noise SNR. The
manifest/cache checks validate the public recording paths and feature shapes.

S2 averages all eligible windows per patient, rather than giving a short and
a long recording equal weight. Unknown CirCor visits are excluded; patients
with conflicting known labels are excluded. Missing BMD-HS annotations are
not converted into negative labels. AR and multiple diagnoses remain eligible
for referral, but not the shared single-class comparison. The latter drops
MVP and renormalises. New regression tests exercise these cases explicitly.

S3 retains the original model, updates BN statistics without label inputs,
and separates adaptation and scoring patients. The complete sweep has four
sites, four batch sizes, five draws and three additional one-sided batch
conditions per site. The collapse guard concerns five-class collapse on the
adaptation batch; it does not guarantee improvement at the referral cutoff.

S4 measures five-class accuracy on a fixed held-out subset of fold 0, with
separate adaptation clips. It covers both noise types at all five SNRs and
all five low-pass cutoffs. These accuracy gains do not establish improvement
in S3's referral macro-F1. Confidence intervals in S3/S4 resample scoring
patients or clips, conditional on the five selected adaptation draws.

## Findings that must remain in the final write-up

- Poor unseen-data generalisation is a measured limitation. S2's model calls
  almost every unseen patient abnormal at the preselected 0.5 cutoff.
- S3's required macro-F1 recovery target was not reached. Increased CirCor
  AUROC does not establish improved referral decisions at that fixed cutoff.
  Clean-data degradation and the unsuccessful settings must stay in the report.
- Temperature below one is not itself an implementation error. The calibration
  report on `main` contains a fitted temperature and lower held-out ECE; the
  roadmap's expectation of temperature above one was a prediction, not a bound.
- A zero feature-vote weight is allowed by the specified weight search. It
  records no measured gain, not proof that the feature file was unused.
- Frequent quality-check failures are reported by the existing public and
  clinic summaries. The roadmap anticipates failures on the new microphone;
  the quality gate must not be relaxed merely to improve the success count.

## Team integration still needed

At `main` revision `47929ce`, the fusion runner covers PhysioNet but not
BMD-HS or adapted variants, and does not write the dashboard's fusion parameter
file. That code is not yet on this `cardio` base. These are separate team tasks;
committing Shivansh's missing outputs does not finish them.

Fusion and S2 also use different CirCor cohorts and aggregation: the fusion
code keeps patients with a positive label on any visit and averages recording
scores; S2 excludes conflicting known labels and weights windows equally.
S3 uses a smaller held-out panel. Their plain-CNN values must not be presented
as though they were measurements on an identical panel. The fusion comparisons
need a shared eligible cohort and documented aggregation before final claims.

The fusion report stores a metadata AUROC gain without a paired gain interval.
The final S5 number check must resolve that and reconcile the final results
tables with committed files. The official output files in this change do not
certify the unfinished dashboard, team comparisons or private clinic reruns.

AI assistance: Codex read the roadmap and feedback, inspected the implementation,
added the missing workflow step and regression tests, and ran the public-data
experiments. Verification uses fresh runs, hashes, manifest joins, deterministic
replays and the project test suite. No private clinic recording was accessed.
