#!/usr/bin/env python3
"""Select path-free, pre-label prediction arrays from frozen research runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def save(path: Path, source: list[Path], **arrays) -> dict:
    ids = np.asarray(arrays["sample_id"]).astype(str)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prediction IDs: {path}")
    for key, values in arrays.items():
        if "path" in key or values.dtype.kind == "O":
            raise ValueError(f"unsafe prediction field: {key}")
        aligned = values.shape[0] == len(ids) or (values.ndim == 3 and values.shape[1] == len(ids))
        if key != "sample_id" and (not aligned or not np.isfinite(values).all()):
            raise ValueError(f"invalid prediction field: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"file": path.name, "rows": len(ids),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_artifact_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in source]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = args.comparisons / "runs"
    destination = args.output / "records/predictions"
    index = []
    full = runs / "QWEN_TOKEN_POOLING_POOLCE_NESTED_TEST_SUPPLEMENTAL_20261025/TEST_PREDICTIONS_PRELABELS.npz"
    f = load(full)
    index.append(save(destination / "full_test_fusion.npz", [full],
                      sample_id=f["sample_id"], probabilities=f["probabilities"],
                      task_members=f["poolce_member_probabilities"],
                      visual_members=f["visual_member_probabilities"]))
    cnn = runs / "FINAL_TEST_MATCHED_RES192_20261026/PREDICTIONS_FROZEN.npz"
    c = load(cnn)
    index.append(save(destination / "full_test_convnext.npz", [cnn],
                      sample_id=c["sample_id"], probabilities=c["convnext_tiny_probabilities"]))
    few = runs / "INSPLAD_STRUCTURED_SFT_KCURVE_20261105_R1/formal"
    for draw in range(3):
        root = few / f"draw_{draw:02d}"
        p5 = root / "K005/S/attempt_01/DEV_P_H_PREDICTIONS_PRELABELS.npz"
        q5 = root / "K005/C0/DEV_PREDICTIONS_PRELABELS.npz"
        a, b = load(p5), load(q5)
        if not np.array_equal(a["sample_id"], b["sample_id"]):
            raise ValueError("K5 branch IDs differ")
        fused = 0.25 * a["P_probabilities"] + 0.75 * b["probabilities"]
        index.append(save(destination / f"fewshot_dev_k5_draw{draw}.npz", [p5, q5],
                          sample_id=a["sample_id"], probabilities=fused,
                          task_probabilities=a["P_probabilities"],
                          reference_probabilities=b["probabilities"]))
        for k in (10, 20):
            path = root / f"K{k:03d}/S_DYN_GATE/DEV_FIXED_AND_DYNAMIC_PRELABELS.npz"
            arr = load(path)
            index.append(save(destination / f"fewshot_dev_k{k}_draw{draw}.npz", [path],
                              sample_id=arr["sample_id"], probabilities=arr["dynamic"],
                              fixed_raw=arr["fixed_raw"], task_probabilities=arr["P_raw"],
                              reference_probabilities=arr["QD_raw"]))
        path = root / "K020/S_DYN_GATE/TEST_FIXED_AND_DYNAMIC_PRELABELS.npz"
        arr = load(path)
        index.append(save(destination / f"fewshot_test_k20_draw{draw}.npz", [path],
                          sample_id=arr["sample_id"], probabilities=arr["dynamic"],
                          fixed_raw=arr["fixed_raw"]))
    (args.output / "records/prediction_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(index), "rows": sum(x["rows"] for x in index)}, indent=2))


if __name__ == "__main__":
    main()
