"""Protect the S2 label exclusions and window-weighted patient scores."""

import csv
import json

import numpy as np

from cardio.eval import run_unseen


def manifest(tmp_path, dataset, patients):
    path = tmp_path / f"{dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "patient_id", "path", "labels"])
        writer.writeheader()
        for index, (patient, labels) in enumerate(patients):
            writer.writerow(dict(dataset=dataset, patient_id=patient,
                                 path=f"{patient}-{index}.wav", labels=json.dumps(labels)))
    return path


def test_circor_drops_unknown_visits_and_conflicting_patients(tmp_path):
    path = manifest(tmp_path, "circor", [
        ("a", {"murmur": "Present"}), ("a", {"murmur": "Unknown"}),
        ("b", {"murmur": "Absent"}), ("b", {"murmur": "Present"}),
        ("c", {"murmur": "Unknown"}), ("d", {"murmur": "Absent"}),
    ])
    rows, targets, shared, counts = run_unseen.patient_cohort("circor", path)
    assert targets == {"a": 1, "d": 0}
    assert [row["patient_id"] for row in rows] == ["a", "d"]
    assert shared == {}
    assert counts["excluded_patients"] == {"conflicting_murmur_labels": 1, "unknown_murmur": 1}


def test_bmdhs_keeps_ar_and_multiple_diseases_only_for_referral(tmp_path):
    def labels(**values):
        return dict(AS=0, AR=0, MS=0, MR=0, N=0) | values

    path = manifest(tmp_path, "bmdhs", [
        ("normal", labels(N=1)), ("as", labels(AS=1)),
        ("ar", labels(AR=1)), ("multiple", labels(AS=1, MR=1)),
        ("missing", labels(N=None)), ("inconsistent", labels(N=1, MS=1)),
    ])
    rows, targets, shared, counts = run_unseen.patient_cohort("bmdhs", path)
    assert targets == {"normal": 0, "as": 1, "ar": 1, "multiple": 1}
    assert shared == {"normal": "Normal", "as": "AS"}
    assert len(rows) == 4
    assert counts["excluded_patients"] == {"missing_disease_labels": 1, "inconsistent_disease_labels": 1}


def test_patient_probability_weights_windows_not_recordings(monkeypatch):
    # One normal window and three AS windows must yield 0.25 normal, not 0.5.
    arrays = [np.eye(5)[[0]], np.eye(5)[[1, 1, 1]], np.eye(5)[[2, 2]], np.empty((0, 5))]
    rows = [dict(patient_id=patient, index=i) for i, patient in enumerate(["a", "a", "b", "c"])]
    monkeypatch.setattr(run_unseen, "load_cached", lambda row: arrays[row["index"]])
    probabilities, recordings, windows = run_unseen.predict_patients(rows, lambda x: x, batch_size=2)
    np.testing.assert_allclose(probabilities["a"], [0.25, 0.75, 0, 0, 0])
    np.testing.assert_allclose(probabilities["b"], [0, 0, 1, 0, 0])
    assert set(probabilities) == {"a", "b"}
    assert (recordings, windows) == (3, 6)


def test_bmdhs_class_test_removes_mvp_before_renormalising(monkeypatch):
    monkeypatch.setattr(run_unseen, "patient_cohort", lambda *args: (
        [], {"a": 0, "b": 1, "c": 1}, {"a": "Normal", "b": "AS"}, {}))
    probabilities = {"a": np.array([.3, .1, 0, 0, .6]),
                     "b": np.array([.1, .4, .1, .2, .2]),
                     "c": np.array([.1, .1, .1, .1, .6])}
    monkeypatch.setattr(run_unseen, "predict_patients", lambda *args: (probabilities, 3, 3))
    calls = []

    def metrics(y, p, patients, names):
        calls.append((y, p, patients))
        return dict.fromkeys(names, {})

    monkeypatch.setattr(run_unseen, "metric_intervals", metrics)
    result = run_unseen.evaluate("bmdhs", None)
    np.testing.assert_allclose(calls[0][1], [.7, .9, .9])
    np.testing.assert_allclose(calls[1][1], [[.75, .25, 0, 0], [.125, .5, .125, .25]])
    np.testing.assert_array_equal(calls[1][0], [0, 1])
    assert calls[1][2] == ["a", "b"]
    assert result["shared_classes"]["patients"] == 2
