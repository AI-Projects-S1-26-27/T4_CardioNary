# S4 Yaseen robustness with the reference model

The float reference checkpoint scored 50 fixed clips from the 197 usable
Yaseen fold-0 recordings. The other clips supplied five batches of 32
adaptation recordings. Each condition used the same batches and scoring clips.
Clean five-class accuracy was 98.0%. Adapted accuracy is the mean of the five
draws. The 95% intervals resample the scoring clips together across both model
variants and all draws. [Reference run](../../reports/heart_robustness_reference.json)

- White noise at 20 dB went from 42.0% plain to 96.0% adapted. At 5 dB it
  went from 40.0% to 58.4%, with a gain interval of −0.4 to 36.8 points.
  At 0 dB both models scored 34.0%.
- Pink noise at 20 dB went from 50.0% to 87.2%. At 0 dB it went from 26.0%
  to 32.4%, with an interval that included zero gain.
- Low-pass accuracy stayed at 98.0% from 600 to 400 Hz and at 96.0% at
  300 Hz. At 200 Hz it went from 80.0% to 93.6%.

Ten of the 15 conditions had a higher point estimate after adaptation. The
other five tied. Seven gain intervals excluded zero. These are top-class
accuracy results. The S3 referral macro-F1 at a fixed probability cutoff
measures a different outcome, so gains here do not establish a referral gain.

Use the confirmed team checkpoint and its paired report for the official
`reports/heart_robustness.json` run. The reproducible commands are in the
[evaluation README](../eval/README.md).

The audit replay reproduced the JSON exactly. The audit also measured each
noise level on a real clip, verified the filter cutoffs passed to the runner,
and recomputed accuracy from all 75 draw counts. All five classes were present
in the scoring panel, and every adaptation batch was disjoint from it.

## Published checkpoint run, 15 November

The updated `cardio` branch includes the published float model and matching
training normalisation report. Their hashes match the reference sweep inputs.
Running `python -m cardio.run_all --only robustness` produced the required
[`reports/heart_robustness.json`](../../reports/heart_robustness.json). Every
condition, per-draw score, confidence interval, and guard decision reproduced
the reference report. This report records the published team checkpoint and
its paired normalisation report hashes.

The full suite on the integrated branch passed 145 tests and 7 subtests, with
4 skips for unavailable caches or an inapplicable image-windowing check.
