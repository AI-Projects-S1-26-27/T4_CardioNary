"""Extract the numeric characteristics of every Kermany (or RSNA) chest X-ray.

For each 96 x 96 image in the cache: the numeric fields of
core.descriptors.vision_descriptors (dense area, asymmetry index, zone percentages,
largest-region fraction, regions detected, mean intensity, texture heterogeneity)
plus the three CLAHE texture numbers of lung.model.clahe. Text fields such as
"pattern" are left out.

Output:
    reports/xray_features_kermany.csv

Usage:
    python -m lung.model.features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.descriptors import vision_descriptors
from lung.model.clahe import CLAHE_COLUMNS, clahe_numbers

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "cache" / "kermany_logmel.npz"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "kermany_manifest.csv"
OUTPUT_PATH = REPO_ROOT / "reports" / "xray_features_kermany.csv"



VISION_COLUMNS = (
    "dense_area_pct", "asymmetry_index", "zone_upper_pct", "zone_middle_pct", "zone_lower_pct",
    "largest_region_fraction", "regions_detected", "mean_intensity", "texture_heterogeneity",
)
FEATURE_COLUMNS = list(VISION_COLUMNS) + list(CLAHE_COLUMNS)


def characteristics(img01: np.ndarray) -> dict[str, float]:
    """The 12 numbers the feature vote uses, for one image in [0, 1].

    This is the single definition: the offline tool and the RSNA run call it too.
    """
    desc = vision_descriptors(img01)
    zones = desc.get("zone_distribution_pct", {})
    row = {
        "dense_area_pct": desc["dense_area_pct"],
        "asymmetry_index": desc["asymmetry_index"],
        "zone_upper_pct": zones.get("upper", 0.0),
        "zone_middle_pct": zones.get("middle", 0.0),
        "zone_lower_pct": zones.get("lower", 0.0),
        "largest_region_fraction": desc["largest_region_fraction"],
        "regions_detected": desc["regions_detected"],
        "mean_intensity": desc["mean_intensity"],
        "texture_heterogeneity": desc["texture_heterogeneity"],
    }
    row.update(clahe_numbers(img01))
    return {k: float(v) for k, v in row.items()}


def extract_features(
    cache_path: Path = CACHE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Compute the characteristics of every image in a cache, in the cache's own order."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}. Run core.preprocess first.")

    print(f"Loading image cache from {cache_path}...")
    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]  # (N, 96, 96) float32 in [0, 1]
    n_images = len(X)
    print(f"Loaded {n_images} images.")

    patient_ids = cache["patient_id"]
    labels = cache["y"]
    folds = cache["fold"]

    filenames, diagnoses = [f"img_{i:05d}" for i in range(n_images)], None
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        if len(man) == n_images:        # a manifest of another length cannot belong to this cache
            if "filename" in man.columns:
                filenames = man["filename"].tolist()
            if "diagnosis" in man.columns:
                diagnoses = man["diagnosis"].tolist()
        else:
            print(f"[warn] manifest has {len(man)} rows but the cache {n_images}; ignoring its names")

    rows = []
    print("Computing characteristics and CLAHE numbers...")
    for i in range(n_images):
        row = {
            "filename": filenames[i],
            "patient_id": str(patient_ids[i]),
            "label": int(labels[i]),
            "fold": int(folds[i]),
            **characteristics(X[i]),
        }
        if diagnoses is not None:
            row["diagnosis"] = diagnoses[i]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Computed characteristics for {len(out_df)} images -> {output_path}")
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="Path to kermany_logmel.npz")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to kermany_manifest.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    extract_features(cache_path=args.cache, manifest_path=args.manifest, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
