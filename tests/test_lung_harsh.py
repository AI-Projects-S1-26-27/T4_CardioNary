"""Harsh's X-ray pieces: CLAHE numbers, the votes, fusion from the fitted file, run_all plan.

Everything runs on made-up data, so it needs no images and no model.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from core.contracts import PatientMetadata
from lung import run_all
from lung.fusion import late_fusion as lf
from lung.fusion.calibrate import expected_calibration_error
from lung.fusion.meta_vote import fit_metadata_vote
from lung.fusion.votes import FEATURE_COLUMNS, fit_feature_vote
from lung.model.clahe import CLAHE_COLUMNS, clahe_numbers, clahe_view, lung_area
from lung.model.features import characteristics


def _img(seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.3, 0.6, 96)[None, :] * np.ones((96, 1))
    return np.clip(base + rng.normal(0, 0.05, (96, 96)), 0, 1).astype(np.float32)


def _features():
    """Twelve made-up characteristics, so the fusion tests do not need OpenCV (the CI has none)."""
    return {c: float(i + 1) for i, c in enumerate(FEATURE_COLUMNS)}


# ------------------------------------------------------------------ CLAHE
def test_clahe_view_is_uint8_96x96_and_does_not_touch_its_input():
    pytest.importorskip("cv2")
    img = _img()
    before = img.copy()
    v = clahe_view(img)
    assert v.shape == (96, 96) and v.dtype == np.uint8
    assert np.array_equal(img, before)


def test_clahe_numbers_are_three_finite_values_in_lung_area():
    pytest.importorskip("cv2")
    nums = clahe_numbers(_img())
    assert tuple(nums) == CLAHE_COLUMNS
    assert all(np.isfinite(v) for v in nums.values())
    assert 0 <= nums["clahe_mean"] <= 255 and nums["clahe_std"] > 0 and nums["clahe_abs_laplacian"] > 0
    assert not lung_area()[:, 48].any() and lung_area()[48, 10]      # mediastinum strip excluded


def test_clahe_numbers_see_texture():
    pytest.importorskip("cv2")
    flat = np.full((96, 96), 0.5, np.float32)
    assert clahe_numbers(flat)["clahe_abs_laplacian"] < clahe_numbers(_img())["clahe_abs_laplacian"]


def test_clahe_rejects_a_colour_image():
    pytest.importorskip("cv2")
    with pytest.raises(ValueError):
        clahe_view(np.zeros((96, 96, 3)))


def test_characteristics_are_the_twelve_feature_columns():
    pytest.importorskip("cv2")
    row = characteristics(_img())
    assert list(row) == FEATURE_COLUMNS and len(row) == 12


# ------------------------------------------------------------------ calibration
def test_ece_defaults_to_15_bins():
    assert inspect.signature(expected_calibration_error).parameters["n_bins"].default == 15


# ------------------------------------------------------------------ fusion from the fitted file
def _params():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    feats = pd.DataFrame({c: rng.normal(y * 0.8, 1.0) for c in FEATURE_COLUMNS})
    feats["label"] = y
    man = pd.DataFrame({"patient_id": np.arange(n).astype(str), "label": y, "fold": np.arange(n) % 5,
                        "age": np.where(rng.random(n) < 0.5, 20 + 50 * y * rng.random(n), np.nan),
                        "sex": rng.choice(["M", "F"], n)})
    return {"T": 0.94, "weights": {"cnn": 0.85, "features": 0.05, "metadata": 0.1},
            "feature_vote": fit_feature_vote(feats), "metadata_vote": fit_metadata_vote(man)}


def test_params_are_plain_json_numbers():
    p = _params()
    json.dumps(p)                      # raises if anything is a numpy object
    assert len(p["feature_vote"]["coef"]) == len(p["feature_vote"]["columns"]) == 12


def test_xray_fusion_uses_every_vote_that_is_present():
    out = lf.xray_fusion(0.8, _features(), PatientMetadata(age_years=60, sex="male"), _params())
    assert out.modality == "xray" and set(out.votes) == {"cnn", "features", "metadata"}
    assert sum(out.weights.values()) == pytest.approx(1.0)
    assert out.p_abnormal == pytest.approx(sum(out.weights[k] * out.votes[k] for k in out.votes))


def test_missing_votes_get_weight_zero_and_the_rest_are_divided_by_their_sum():
    p = _params()
    out = lf.xray_fusion(0.7, None, None, p)
    assert out.votes == {"cnn": 0.7} and out.weights == {"cnn": 1.0}
    out = lf.xray_fusion(0.7, _features(), None, p)
    assert set(out.weights) == {"cnn", "features"}
    assert out.weights["cnn"] == pytest.approx(0.85 / 0.90)


def test_metadata_weight_is_clamped_to_the_cap_when_other_votes_drop_out():
    p = _params()
    p["weights"] = {"cnn": 0.0, "features": 0.7, "metadata": 0.3}
    out = lf.xray_fusion(0.7, None, PatientMetadata(sex="female"), p)     # features dropped
    assert out.weights["metadata"] <= 0.3 + 1e-9 and sum(out.weights.values()) == pytest.approx(1.0)


def test_symptoms_alone_are_not_a_metadata_vote():
    out = lf.xray_fusion(0.5, None, PatientMetadata(symptoms=frozenset({"cough"})), _params())
    assert "metadata" not in out.votes


def test_missing_age_uses_the_fill():
    mv = _params()["metadata_vote"]
    only_sex = lf.metadata_vote_probability(PatientMetadata(sex="male"), mv)
    filled = lf.metadata_vote_probability(PatientMetadata(age_years=mv["age_fill_years"], sex="male"), mv)
    assert only_sex == pytest.approx(filled)
    assert 0 < lf.metadata_vote_probability(PatientMetadata(age_years=40), mv) < 1


def test_a_single_missing_characteristic_is_filled_with_its_median():
    vote = _params()["feature_vote"]
    full = _features()
    fill = vote["fill"][vote["columns"].index("clahe_std")]
    a = lf.feature_vote_probability(dict(full, clahe_std=float("nan")), vote)
    b = lf.feature_vote_probability(dict(full, clahe_std=fill), vote)
    assert a == pytest.approx(b)


def test_invalid_cnn_probability_is_refused():
    with pytest.raises(ValueError):
        lf.xray_fusion(1.4, None, None, _params())


def test_committed_params_file_gives_a_valid_answer():
    if not lf.PARAMS_JSON.exists():
        pytest.skip("reports/xray_fusion_params.json not generated yet")
    p = json.loads(lf.PARAMS_JSON.read_text())
    out = lf.xray_fusion(0.5, _features(), PatientMetadata(age_years=50, sex="female"), p)
    assert 0 <= out.p_abnormal <= 1 and out.weights["metadata"] <= 0.3 + 1e-9


def test_fused_kermany_is_calibrated_per_fold_and_aligned(tmp_path):
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    oof = pd.DataFrame({"index": np.arange(n), "patient_id": (np.arange(n) // 2).astype(str),
                        "fold": np.arange(n) % 5, "label": y,
                        "p_pneumonia": np.clip(0.1 + 0.8 * y + rng.normal(0, 0.1, n), 0.01, 0.99)})
    plr = pd.DataFrame({"patient_id": oof.patient_id, "label": y,
                        "pLR": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, n), 0.01, 0.99)})
    out = lf.fuse_dataset(oof, plr, None, "t", tmp_path / "f.csv", tmp_path / "f.json")
    assert len(out) == n and all(o.modality == "xray" for o in out)
    s = json.loads((tmp_path / "f.json").read_text())
    assert s["votes_used"] == ["cnn", "features"] and len(s["per_fold"]) == 5
    with pytest.raises(ValueError):                                   # shuffled vote file is refused
        lf.fuse_dataset(oof, plr.sample(frac=1, random_state=0).reset_index(drop=True), None, "t",
                        tmp_path / "g.csv", tmp_path / "g.json")


# ------------------------------------------------------------------ run_all
def test_run_all_steps_are_unique_and_their_code_exists():
    names = [s.name for s in run_all.STEPS]
    assert len(names) == len(set(names))
    for s in run_all.STEPS:
        assert s.code.exists(), f"{s.module}: no such file"
        assert s.writes and all(w.startswith(("reports/", "data/cache/")) for w in s.writes)


def test_fusion_params_come_after_rsna_fusion_and_every_xray_json_has_a_step():
    names = [s.name for s in run_all.STEPS]
    assert names.index("fusion_params") > names.index("fusion_rsna") > names.index("rsna_adapted")
    written = {w for s in run_all.STEPS for w in s.writes}
    for f in ("xray_fusion.json", "xray_fusion_params.json", "xray_tta.json", "xray_robustness.json",
              "xray_edge_budget.json", "xray_timing.json", "xray_rsna.json", "xray_calibration_kermany.json"):
        assert f"reports/{f}" in written


def test_only_from_and_unknown_step():
    assert [s.name for s, _ in run_all.plan(only=["votes"])] == ["votes"]
    assert run_all.plan(start="fusion_rsna")[0][0].name == "fusion_rsna"
    with pytest.raises(ValueError):
        run_all.plan(only=["nope"])


# ------------------------------------------------------------------ RSNA metadata
def test_rsna_age_is_read_with_or_without_a_unit():
    pytest.importorskip("pydicom")
    from lung.data.build_rsna import parse_age
    # The RSNA files hold bare numbers ("51"); the DICOM standard form is "051Y".
    assert parse_age("51") == 51.0
    assert parse_age("057Y") == 57.0
    assert parse_age("006M") == pytest.approx(0.5)
    assert np.isnan(parse_age("")) and np.isnan(parse_age(None)) and np.isnan(parse_age("abc"))


def test_metadata_vote_is_out_of_fold_and_learns_age():
    from lung.fusion.meta_vote import oof_metadata_vote
    rng = np.random.default_rng(0)
    n = 1000
    age = rng.uniform(20, 90, n)
    y = (rng.uniform(size=n) < (age - 20) / 70).astype(int)   # older -> more opacity
    df = pd.DataFrame({"fold": np.arange(n) % 5, "label": y, "age": age,
                       "sex": rng.choice(["M", "F"], n)})
    df.loc[::17, "age"] = np.nan                               # some ages missing
    p = oof_metadata_vote(df)
    from sklearn.metrics import roc_auc_score
    assert p.shape == (n,) and np.all((p > 0) & (p < 1))
    assert roc_auc_score(y, p) > 0.7
    # changing the labels of fold 0 must not move fold 0's own predictions
    df2 = df.copy()
    df2.loc[df2.fold == 0, "label"] = 1 - df2.loc[df2.fold == 0, "label"]
    assert np.allclose(oof_metadata_vote(df2)[df.fold == 0], p[df.fold == 0])
