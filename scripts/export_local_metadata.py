#!/usr/bin/env python3
"""One-time export from the research workspace into path-free release records.

This script is intentionally separate from the public data preparation command:
readers need only the committed release manifest and the upstream InsPLAD archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifests", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--result-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    with (args.dataset_manifests / "image_group_map_v1.csv").open(newline="", encoding="utf-8") as stream:
        group_map = {r["sample_id"]: r for r in csv.DictReader(stream)}
    with (args.dataset_manifests / "master_manifest.csv").open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    rows = []
    for r in original:
        path = r["image_path"].replace("\\", "/")
        marker = "/defect_supervised/"
        if path.count(marker) != 1:
            raise ValueError(f"cannot relativize sample {r['sample_id']}")
        group = group_map[r["sample_id"]]
        if group["unified_group_id"] != r["unified_group_id"]:
            raise ValueError("group map and manifest disagree")
        rows.append({
            "sample_id": r["sample_id"], "relative_path": path.split(marker, 1)[1],
            "leaf_id": r["leaf_id"], "leaf_name": r["leaf_name"],
            "group_id": r["unified_group_id"], "split": r["split"],
            "sha256": group["sha256"], "source_id": group["source_id"],
            "group_reason": group["group_reason"],
        })
    rows.sort(key=lambda r: r["sample_id"])
    write_csv(output / "data/manifests/insplad_clean_v1.csv", rows,
              ["sample_id", "relative_path", "leaf_id", "leaf_name", "group_id", "split",
               "sha256", "source_id", "group_reason"])

    fold = args.comparisons / "runs/CLIP_VIT_B32_TRAIN_OOF_20261001T133608Z/TRAIN_ID_GROUP_FOLD_MANIFEST.jsonl"
    fold_rows = [{k: r[k] for k in ("sample_id", "group_id", "outer_fold")} for r in read_jsonl(fold)]
    write_csv(output / "data/provenance/historical_outer_folds.csv", fold_rows,
              ["sample_id", "group_id", "outer_fold"])

    member_path = args.comparisons / "runs/FINAL_TEST_MATCHED_RES192_20261026/MEMBERS.json"
    members = json.loads(member_path.read_text(encoding="utf-8"))["members"]
    member_rows = [{"member_id": m["member_id"], "sample_id": sid}
                   for m in members for sid in m["train_ids"]]
    write_csv(output / "data/provenance/full_members.csv", member_rows,
              ["member_id", "sample_id"])

    few = args.comparisons / "runs/INSPLAD_STRUCTURED_SFT_KCURVE_20261105_R1/prepared"
    support_rows = []
    for draw in range(3):
        for k in (5, 10, 20):
            path = few / f"draw_{draw:02d}/K{k:03d}/TRAIN_ROWS.jsonl"
            for r in read_jsonl(path):
                support_rows.append({"draw": draw, "k": k, "sample_id": r["sample_id"],
                                     "group_id": r["group_id"], "leaf_id": r["label_id"]})
    write_csv(output / "data/provenance/fewshot_supports.csv", support_rows,
              ["draw", "k", "sample_id", "group_id", "leaf_id"])
    oof_rows = []
    for k in (10, 20):
        for draw in range(3):
            for fold in range(2):
                directory = few / f"dynamic_gate_oof/K{k:03d}/draw_{draw:02d}/fold_{fold:02d}"
                for role, filename in (("train", "TRAIN_ROWS.jsonl"),
                                       ("validation", "VALIDATION_ROWS_NO_LABELS.jsonl")):
                    for r in read_jsonl(directory / filename):
                        oof_rows.append({"draw": draw, "k": k, "fold": fold,
                                         "role": role, "sample_id": r["sample_id"],
                                         "group_id": r["group_id"]})
    write_csv(output / "data/provenance/fewshot_oof_folds.csv", oof_rows,
              ["draw", "k", "fold", "role", "sample_id", "group_id"])
    query_rows = []
    for name, cohort in (("DEV_QUERIES_NO_LABELS.jsonl", "development"),
                         ("TEST_QUERIES_NO_LABELS.jsonl", "fewshot_test")):
        for r in read_jsonl(few / name):
            query_rows.append({"cohort": cohort, "sample_id": r["sample_id"], "group_id": r["group_id"]})
    write_csv(output / "data/provenance/fewshot_queries.csv", query_rows,
              ["cohort", "sample_id", "group_id"])

    bindings = json.loads(args.result_bindings.read_text(encoding="utf-8"))
    clean = []
    for entry in bindings:
        e = {key: value for key, value in entry.items() if key not in {"source", "path"}}
        source = Path(entry.get("source", entry.get("path", "")))
        e["historical_artifact"] = "/".join(source.parts[-2:])
        clean.append(e)
    target = output / "records/historical_result_bindings.json"
    target.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256((output / "data/manifests/insplad_clean_v1.csv").read_bytes()).hexdigest()
    print(json.dumps({"manifest_rows": len(rows), "fold_rows": len(fold_rows),
                      "member_rows": len(member_rows), "support_rows": len(support_rows),
                      "query_rows": len(query_rows), "oof_rows": len(oof_rows),
                      "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
