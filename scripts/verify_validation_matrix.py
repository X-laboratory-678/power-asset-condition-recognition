#!/usr/bin/env python3
"""Recheck public functional-run predictions, hashes, metrics and fusion."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from power_asset_condition.data import load_manifest
from power_asset_condition.fusion import EntropyGate, fixed_fewshot, full_ensemble
from power_asset_condition.metrics import check_probabilities, paired_score
from power_asset_condition.teacher import SYSTEM as TEACHER_SYSTEM, USER as TEACHER_USER, TEACHER_IMAGE_MAX_PIXELS

ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    matrix = json.loads((ROOT / "records/validation_matrix.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/manifests/insplad_clean_v1.csv"
    if _digest(manifest_path) != matrix["manifest_sha256"]:
        raise ValueError("manifest digest differs from validation record")
    if _digest(ROOT / "configs/paper.json") != matrix["config_sha256"]:
        raise ValueError("configuration digest differs from validation record")
    prompt_record = matrix["teacher_generation"]
    if (prompt_record["system_prompt_sha256"] != hashlib.sha256(TEACHER_SYSTEM.encode()).hexdigest()
            or prompt_record["user_prompt_sha256"] != hashlib.sha256(TEACHER_USER.encode()).hexdigest()
            or matrix["image_limits"]["qwen3_teacher_max_pixels"] != TEACHER_IMAGE_MAX_PIXELS):
        raise ValueError("teacher prompt or image limit differs from validation record")
    config = json.loads((ROOT / "configs/paper.json").read_text(encoding="utf-8"))
    if matrix["image_limits"]["qwen3_training_max_pixels"] != config["image_max_pixels"]:
        raise ValueError("training image limit differs from configuration")
    rows = load_manifest(manifest_path)
    by_id = {row["sample_id"]: row for row in rows}
    development = {row["sample_id"] for row in _csv(ROOT / "data/provenance/fewshot_queries.csv")
                   if row["cohort"] == "development"}
    development_rows = [row for row in _csv(ROOT / "data/provenance/fewshot_queries.csv")
                        if row["cohort"] == "development"]
    supports = _csv(ROOT / "data/provenance/fewshot_supports.csv")
    full_members = _csv(ROOT / "data/provenance/full_members.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    for cache in matrix.get("teacher_sources", []):
        expected_ids = [row["sample_id"] for row in supports
                        if int(row["draw"]) == cache["draw"] and int(row["k"]) == cache["k"]]
        cache_ids = cache.get("sample_ids", [])
        if (cache["rows"] != len(expected_ids) or len(cache_ids) != len(expected_ids)
                or set(cache_ids) != set(expected_ids)):
            raise ValueError(f"teacher cache membership differs: K={cache['k']}")

    def first_per_class(sequence: list[dict], count: int) -> list[str]:
        selected, counts = [], Counter()
        for row in sequence:
            label = int(row["leaf_id"])
            if counts[label] < count:
                selected.append(row["sample_id"])
                counts[label] += 1
        return selected

    expected_queries = first_per_class(
        [{**row, "leaf_id": by_id[row["sample_id"]]["leaf_id"]} for row in development_rows], 1)
    checked = []
    for run in matrix["runs"]:
        if run["status"] != "PASS":
            raise ValueError(f"validation path has not passed: {run['id']}")
        relative = Path(run["prediction_file"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("records", "predictions"):
            raise ValueError("unsafe prediction file path")
        prediction_path = ROOT / relative
        if _digest(prediction_path) != run["sha256"]:
            raise ValueError(f"prediction digest differs: {run['id']}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            ids = archive["sample_id"].astype(str).tolist()
            probabilities = check_probabilities(archive["probabilities"])
            if len(ids) != 11 or len(set(ids)) != 11 or ids != expected_queries or not set(ids) <= development:
                raise ValueError(f"not 11 distinct Development queries: {run['id']}")
            if Counter(by_id[sid]["leaf_id"] for sid in ids) != {label: 1 for label in range(11)}:
                raise ValueError(f"query class distribution differs: {run['id']}")
            fit_sets = (list(run["fit_sample_ids_by_member"].values())
                        if "fit_sample_ids_by_member" in run else [run["fit_sample_ids"]])
            if run["id"] == "full":
                for member, fit_ids in run["fit_sample_ids_by_member"].items():
                    member_rows = [row for row in full_members if row["member_id"] == member]
                    expected_fit = first_per_class(
                        [{**row, "leaf_id": by_id[row["sample_id"]]["leaf_id"]} for row in member_rows], 2)
                    if fit_ids != expected_fit:
                        raise ValueError(f"full member fit membership differs: {member}")
            elif run["id"] in ("k5", "k10", "k20", "ridge"):
                support_rows = [row for row in supports
                                if int(row["draw"]) == run.get("draw", 0)
                                and int(row["k"]) == (5 if run["id"] == "ridge" else int(run["k"]))]
                expected_fit = [row["sample_id"] for row in support_rows]
                if run["fit_sample_ids"] != expected_fit:
                    raise ValueError(f"few-shot fit membership differs: {run['id']}")
                if run["id"] in ("k10", "k20"):
                    fold_rows = [row for row in _csv(ROOT / "data/provenance/fewshot_oof_folds.csv")
                                 if int(row["draw"]) == run["draw"] and int(row["k"]) == run["k"]]
                    covered = []
                    for fold in range(2):
                        entries = [row for row in fold_rows if int(row["fold"]) == fold]
                        fold_fit = [row["sample_id"] for row in entries if row["role"] == "train"]
                        fold_valid = [row["sample_id"] for row in entries if row["role"] == "validation"]
                        if (set(fold_fit) | set(fold_valid) != set(expected_fit)
                                or set(fold_fit) & set(fold_valid)):
                            raise ValueError(f"support OOF fold does not partition supports: {run['id']}/{fold}")
                        fit_groups = {by_id[sid]["group_id"] for sid in fold_fit}
                        valid_groups = {by_id[sid]["group_id"] for sid in fold_valid}
                        if fit_groups & valid_groups:
                            raise ValueError(f"support OOF fold crosses groups: {run['id']}/{fold}")
                        report = run["training_reports"]["support_oof"][f"fold_{fold}"]
                        if (report["fit_rows"] != len(fold_fit)
                                or report["query_rows"] != len(fold_valid)):
                            raise ValueError(f"support OOF run size differs from fold record: {run['id']}/{fold}")
                        covered.extend(fold_valid)
                    if len(covered) != len(expected_fit) or set(covered) != set(expected_fit):
                        raise ValueError(f"support OOF validation coverage is incomplete: {run['id']}")
            else:
                expected_fit = first_per_class(train_rows, 4)
                if run["fit_sample_ids"] != expected_fit:
                    raise ValueError(f"baseline fit membership differs: {run['id']}")
            query_groups = {by_id[sid]["group_id"] for sid in ids}
            expected_per_class = run["fit_rows"] // 11
            for fit_ids in fit_sets:
                if len(fit_ids) != run["fit_rows"] or len(set(fit_ids)) != len(fit_ids):
                    raise ValueError(f"fit IDs do not match recorded fit count: {run['id']}")
                if not set(fit_ids) <= by_id.keys():
                    raise ValueError(f"fit IDs are absent from manifest: {run['id']}")
                if query_groups & {by_id[sid]["group_id"] for sid in fit_ids}:
                    raise ValueError(f"fit and query groups overlap: {run['id']}")
                if Counter(by_id[sid]["leaf_id"] for sid in fit_ids) != {
                        label: expected_per_class for label in range(11)}:
                    raise ValueError(f"fit class counts differ: {run['id']}")
            mode = run.get("fusion", {}).get("mode")
            if mode == "full_ensemble":
                expected = full_ensemble(archive["task_members"], archive["reference_members"])
            elif mode == "fixed_raw":
                expected = fixed_fewshot(archive["task_probabilities"], archive["reference_probabilities"])
                if not np.allclose(expected, archive["fixed_raw"], atol=1e-8):
                    raise ValueError(f"fixed branch differs: {run['id']}")
            elif mode == "entropy_gate":
                oof_path = ROOT / run["support_oof_file"]
                if _digest(oof_path) != run["support_oof_sha256"]:
                    raise ValueError(f"support OOF digest differs: {run['id']}")
                with np.load(oof_path, allow_pickle=False) as oof:
                    support_ids = oof["sample_id"].astype(str).tolist()
                    recorded_ids = [entry["sample_id"] for entry in supports
                                    if int(entry["draw"]) == run["draw"] and int(entry["k"]) == run["k"]]
                    if support_ids != recorded_ids:
                        raise ValueError(f"support OOF order differs: {run['id']}")
                    labels = np.array([by_id[sid]["leaf_id"] for sid in support_ids], dtype=int)
                    gate = EntropyGate.fit(oof["task_probabilities"],
                                           oof["reference_probabilities"], labels)
                recorded_gate = run["fusion"]
                for key in ("task_temperature", "reference_temperature", "beta"):
                    if not np.isclose(getattr(gate, key), recorded_gate[key], atol=1e-8):
                        raise ValueError(f"support gate differs: {run['id']}: {key}")
                expected = gate.predict(archive["task_probabilities"],
                                        archive["reference_probabilities"])
                if not np.allclose(fixed_fewshot(archive["task_probabilities"],
                                                    archive["reference_probabilities"]),
                                   archive["fixed_raw"], atol=1e-8):
                    raise ValueError(f"fixed control differs: {run['id']}")
            elif mode is None:
                expected = probabilities
            else:
                raise ValueError(f"unknown fusion mode: {mode}")
            if not np.allclose(expected, probabilities, atol=1e-8):
                raise ValueError(f"fusion output differs: {run['id']}")
            metrics = paired_score({sid: row["leaf_id"] for sid, row in by_id.items()},
                                   ids, probabilities)
            if metrics != run["metrics"]:
                raise ValueError(f"saved metrics differ: {run['id']}")
        checked.append(run["id"])
    return {"status": "PASS", "paths": checked, "count": len(checked)}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
