#!/usr/bin/env python3
"""Recalculate archived historical scores from path-free frozen predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from power_asset_condition.data import load_manifest, sha256_file
from power_asset_condition.metrics import paired_score


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/manifests/insplad_clean_v1.csv"
    label_by_id = {r["sample_id"]: r["leaf_id"] for r in load_manifest(manifest)}
    records = {}
    index = json.loads((root / "records/prediction_index.json").read_text(encoding="utf-8"))
    for entry in sorted(index, key=lambda item: item["file"]):
        path = root / "records/predictions" / entry["file"]
        with np.load(path, allow_pickle=False) as packed:
            ids = packed["sample_id"].astype(str).tolist()
            probabilities = packed["probabilities"]
        records[path.stem] = {"metrics": paired_score(label_by_id, ids, probabilities),
                              "prediction_sha256": sha256_file(path)}
    means = {}
    for cohort in ("fewshot_dev_k5", "fewshot_dev_k10", "fewshot_dev_k20", "fewshot_test_k20"):
        draws = [records[f"{cohort}_draw{draw}"]["metrics"] for draw in range(3)]
        means[cohort] = {key: float(np.mean([d[key] for d in draws]))
                         for key in ("accuracy", "macro_f1", "balanced_accuracy")}
    output = {"status": "RECOMPUTED_FROM_HISTORICAL_FROZEN_PREDICTIONS",
              "manifest_sha256": sha256_file(manifest), "records": records, "draw_means": means}
    path = root / "records/historical_metrics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "draw_means": means}, indent=2))


if __name__ == "__main__":
    main()
