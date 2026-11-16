"""Required S2-S4 results must be committed, complete and internally consistent.

These checks deliberately fail for missing reports. No raw data or inference is
needed in CI; model/report hashes and manifest joins detect stale deliverables.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cardio.eval.run_unseen import patient_cohort
from cardio.model.final_model import CLASSES, STEM

ROOT = Path(__file__).resolve().parents[1]


def report(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def check_interval(block, *, bounded=True):
    assert {"estimate", "lower_95", "upper_95", "valid_resamples"} <= block.keys()
    assert 0 <= block["valid_resamples"] <= 1000
    if block["valid_resamples"] == 0:
        assert block["lower_95"] is block["upper_95"] is None
        return
    low, high = block["lower_95"], block["upper_95"]
    assert np.isfinite([low, high]).all() and low <= high
    if block["estimate"] is not None:
        assert np.isfinite(block["estimate"])
    if bounded:
        assert 0 <= low <= high <= 1
        assert block["estimate"] is not None and 0 <= block["estimate"] <= 1
    # A percentile bootstrap interval need not contain the original estimate.


@pytest.mark.parametrize("name", ["heart_unseen.json", "heart_tta.json", "heart_robustness.json"])
def test_official_reports_use_the_published_model_and_normalisation(name):
    result = report(name)
    assert result["model_status"] in {"supplied_model", "confirmed_team_checkpoint"}
    assert result["classes"] == CLASSES
    for field, path in [
        ("model_sha256", ROOT / "models" / f"{STEM}_float.keras"),
        ("normalization_report_sha256", ROOT / "reports" / f"quantization_{STEM}.json"),
    ]:
        assert result[field] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("dataset,filename", [("bmdhs", "bmdhs"), ("circor", "physionet")])
def test_unseen_characteristics_cover_the_manifest_once(dataset, filename):
    with (ROOT / f"data/manifests/{dataset}_manifest.csv").open(newline="", encoding="utf-8") as stream:
        expected = {(row["path"], row["patient_id"]) for row in csv.DictReader(stream)}
    with (ROOT / f"reports/heart_features_{filename}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected)
    assert {(row["path"], row["patient_id"]) for row in rows} == expected
    for row in rows:
        assert {"heart_rate_bpm", "regularity", "segmentation_ok", "spectral_centroid_hz"} <= row.keys()
        for key, value in row.items():
            if key not in {"patient_id", "path"} and value:
                assert np.isfinite(float(value))


def test_unseen_cohorts_and_required_intervals():
    result = report("heart_unseen.json")
    assert set(result["datasets"]) == {"bmdhs", "circor"}
    for dataset, block in result["datasets"].items():
        rows, targets, shared, counts = patient_cohort(dataset, ROOT / f"data/manifests/{dataset}_manifest.csv")
        for key, value in counts.items():
            assert block[key] == value
        referral = block["refer_or_not"]
        assert referral["patients"] + block["patients_without_windows"] == len(targets)
        assert 0 <= referral["positive_patients"] <= referral["patients"]
        assert 0 <= referral["predicted_refer_patients"] <= referral["patients"]
        assert block["recordings_scored"] <= len(rows)
        assert set(referral["metrics"]) == {"auroc", "macro_f1", "sensitivity_at_90_specificity", "ece"}
        for value in referral["metrics"].values():
            assert value["resamples"] == 1000
            check_interval(value)
        if dataset == "bmdhs":
            classes = block["shared_classes"]
            assert set(classes["per_class_recall"]) == {"Normal", "AS", "MS", "MR"}
            assert classes["patients"] <= len(shared)
            assert sum(classes["class_support"].values()) == classes["patients"]
            for value in [classes["macro_f1"], *classes["per_class_recall"].values()]:
                assert value["resamples"] == 1000
                check_interval(value)


def test_tta_includes_every_size_stress_batch_draw_and_interval():
    result = report("heart_tta.json")
    assert result["pilot"] is False
    assert set(result["sites"]) == {"bmdhs", "circor", "white_noise", "low_pass"}
    expected = {(n, None) for n in (8, 16, 32, 64)} | {(32, f) for f in (.1, .5, .9)}
    for site in result["sites"].values():
        assert len(site["conditions"]) == len(expected)
        assert {(c["adaptation_recordings"], c["target_abnormal_fraction"]) for c in site["conditions"]} == expected
        assert 0 < site["score_positive_patients"] < site["score_patients"]
        for condition in site["conditions"]:
            draws = condition["draws"]
            assert len(draws) == 5
            for draw in draws:
                n = condition["adaptation_recordings"]
                assert 0 < draw["adaptation_patients"] <= n <= draw["adaptation_windows"]
                assert draw["seconds_to_adapt_with_guard"] > 0
                fraction = condition["target_abnormal_fraction"]
                if fraction is not None:
                    assert draw["abnormal_recordings"] == round(n * fraction)
            for metric in ("macro_f1", "auroc"):
                values = condition["shifted"][metric]
                assert set(values) == {"clean", "plain", "adapted", "gain", "recovery"}
                for name, block in values.items():
                    check_interval(block, bounded=name not in {"gain", "recovery"})
                after = np.mean([draw["shifted_after"][metric] for draw in draws])
                assert values["adapted"]["estimate"] == pytest.approx(after)
                gain = after - values["plain"]["estimate"]
                assert values["gain"]["estimate"] == pytest.approx(gain)
                gap = values["clean"]["estimate"] - values["plain"]["estimate"]
                if gap > 0:
                    assert values["recovery"]["estimate"] == pytest.approx(gain / gap)
                else:
                    assert values["recovery"]["estimate"] is None
                check_interval(condition["clean_after_adaptation"][metric])


def test_robustness_covers_all_15_conditions_and_reconciles_draw_counts():
    result = report("heart_robustness.json")
    expected = {(kind, n) for kind in ("white_noise", "pink_noise") for n in (20, 15, 10, 5, 0)}
    expected |= {("low_pass", n) for n in (600, 500, 400, 300, 200)}
    assert len(result["conditions"]) == len(expected)
    assert {(c["corruption"], c.get("snr_db", c.get("cutoff_hz"))) for c in result["conditions"]} == expected
    assert result["fold"] == 0
    assert result["scored_clips"] + result["adaptation_pool_clips"] == result["eligible_clips"]
    assert result["adaptation_recordings_per_draw"] == 32
    check_interval(result["clean_accuracy"])
    for condition in result["conditions"]:
        assert len(condition["draws"]) == result["draws_per_condition"] == 5
        n = result["scored_clips"]
        plain = condition["plain_correct_clips"] / n
        adapted = np.mean([draw["correct_clips"] / n for draw in condition["draws"]])
        for name, value in [("plain", plain), ("adapted", adapted), ("gain", adapted - plain)]:
            block = condition["accuracy"][name]
            assert block["estimate"] == pytest.approx(value)
            check_interval(block, bounded=name != "gain")
