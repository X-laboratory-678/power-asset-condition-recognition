#!/usr/bin/env python3
"""Check frozen metadata, group boundaries, hashes and saved historical scores."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from power_asset_condition.data import assert_group_disjoint, load_manifest, select_rows, validate_historical
from power_asset_condition.metrics import paired_score

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name: str) -> list[dict]:
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check() -> dict:
    manifest_path = ROOT / "data/manifests/insplad_clean_v1.csv"
    rows = load_manifest(manifest_path)
    historical = validate_historical(rows)
    by_id = {r["sample_id"]: r for r in rows}
    assert len(by_id) == 11525
    assert {r["leaf_id"] for r in rows} == set(range(11))
    supports = read_csv("data/provenance/fewshot_supports.csv")
    queries = read_csv("data/provenance/fewshot_queries.csv")
    for draw in range(3):
        for k in (5, 10, 20):
            selected = [r for r in supports if int(r["draw"]) == draw and int(r["k"]) == k]
            assert len(selected) == 11 * k
            assert Counter(int(r["leaf_id"]) for r in selected) == {i: k for i in range(11)}
            fit = select_rows(rows, [r["sample_id"] for r in selected])
            assert all(r["group_id"] == by_id[r["sample_id"]]["group_id"] for r in selected)
            for cohort in ("development", "fewshot_test"):
                query = select_rows(rows, [r["sample_id"] for r in queries if r["cohort"] == cohort])
                assert_group_disjoint(fit, query)
    oof = read_csv("data/provenance/fewshot_oof_folds.csv")
    for draw in range(3):
        for k in (10, 20):
            support_ids = {r["sample_id"] for r in supports if int(r["draw"]) == draw and int(r["k"]) == k}
            validation_ids = []
            for fold in range(2):
                part = [r for r in oof if int(r["draw"]) == draw and int(r["k"]) == k and int(r["fold"]) == fold]
                tr = {r["sample_id"] for r in part if r["role"] == "train"}
                va = {r["sample_id"] for r in part if r["role"] == "validation"}
                assert tr.isdisjoint(va) and tr | va == support_ids
                assert {by_id[i]["group_id"] for i in tr}.isdisjoint({by_id[i]["group_id"] for i in va})
                assert {by_id[i]["leaf_id"] for i in tr} == set(range(11))
                validation_ids.extend(va)
            assert len(validation_ids) == len(support_ids) and set(validation_ids) == support_ids
    saved = json.loads((ROOT / "records/historical_metrics.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == saved["manifest_sha256"]
    index = json.loads((ROOT / "records/prediction_index.json").read_text(encoding="utf-8"))
    for entry in index:
        path = ROOT / "records/predictions" / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        key = path.stem
        with np.load(path, allow_pickle=False) as z:
            result = paired_score({sid: r["leaf_id"] for sid, r in by_id.items()},
                                  z["sample_id"].astype(str).tolist(), z["probabilities"])
        assert result == saved["records"][key]["metrics"]
    return {"manifest_rows": historical["rows"], "classes": 11, "support_draws": 9,
            "oof_fold_pairs": 6, "prediction_archives": len(index),
            "manifest_sha256": digest, "status": "PASS"}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
