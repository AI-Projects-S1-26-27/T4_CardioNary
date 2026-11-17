"""Tests for RSNA manifest metadata parsing."""

import math

import pytest

from lung.data import build_rsna


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("057Y", 57.0),
        ("57", 57.0),
        (57, 57.0),
        ("006M", 0.5),
        ("002W", 2 / 52),
        ("010D", 10 / 365),
    ],
)
def test_parse_age_supported_formats(value, expected):
    assert build_rsna.parse_age(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "unknown", "57 years", "1000Y"])
def test_parse_age_returns_nan_for_missing_or_invalid_values(value):
    assert math.isnan(build_rsna.parse_age(value))


def test_read_dicom_metadata_reads_header_without_pixels(monkeypatch, tmp_path):
    class Dataset:
        PatientAge = "57"
        PatientSex = "F"
        ViewPosition = "AP"

    observed = {}

    def fake_dcmread(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return Dataset()

    monkeypatch.setattr(build_rsna.pydicom, "dcmread", fake_dcmread)
    result = build_rsna.read_dicom_metadata(tmp_path / "case.dcm")

    assert result == {"age": 57.0, "sex": "F", "view_position": "AP"}
    assert observed["stop_before_pixels"] is True
