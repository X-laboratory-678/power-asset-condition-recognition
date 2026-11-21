"""Dataset interface: stable IDs and relative paths, never machine paths."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import CLASS_ORDER

EXPECTED_SPLITS = {"train": 8039, "val": 1744, "test": 1742}
REQUIRED = {"sample_id", "relative_path", "leaf_id", "group_id", "split", "sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, data_root: Path | None = None, verify_images: bool = False) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest missing columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    ids = set()
    for row in rows:
        sid = row["sample_id"]
        if sid in ids:
            raise ValueError(f"duplicate sample_id: {sid}")
        ids.add(sid)
        label = int(row["leaf_id"])
        if label not in range(len(CLASS_ORDER)) or row.get("leaf_name", CLASS_ORDER[label]) != CLASS_ORDER[label]:
            raise ValueError(f"noncanonical class for {sid}")
        if row["split"] not in EXPECTED_SPLITS:
            raise ValueError(f"unknown split for {sid}")
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"unsafe path for {sid}: {rel}")
        if len(row["sha256"]) != 64:
            raise ValueError(f"missing image hash for {sid}")
        row["leaf_id"] = label
        if data_root is not None:
            row["image_path"] = str(data_root / rel)
            if verify_images:
                image = data_root / rel
                if not image.is_file() or sha256_file(image) != row["sha256"]:
                    raise ValueError(f"missing or changed image: {sid}")
    return rows


def validate_historical(rows: list[dict], strict_counts: bool = True) -> dict:
    counts = Counter(row["split"] for row in rows)
    if strict_counts and counts != EXPECTED_SPLITS:
        raise ValueError(f"historical partition counts changed: {counts}")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split"])
    crossed = {group: sorted(parts) for group, parts in groups.items() if len(parts) > 1}
    if crossed:
        raise ValueError(f"{len(crossed)} historical groups cross partitions; first={next(iter(crossed.items()))}")
    return {"rows": len(rows), "split_counts": dict(counts), "groups": len(groups), "cross_split_groups": 0}


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["sample_id"]: row for row in rows}


def select_rows(rows: list[dict], ids: list[str], *, require_all_classes: bool = True) -> list[dict]:
    index = rows_by_id(rows)
    if len(ids) != len(set(ids)) or any(sid not in index for sid in ids):
        raise ValueError("selection has duplicate or unknown sample IDs")
    selected = [index[sid] for sid in ids]
    if require_all_classes and set(row["leaf_id"] for row in selected) != set(range(11)):
        raise ValueError("selection does not cover all 11 classes")
    return selected


def assert_group_disjoint(fit: list[dict], query: list[dict], group_key: str = "group_id") -> None:
    fit_ids = {r["sample_id"] for r in fit}
    query_ids = {r["sample_id"] for r in query}
    if len(fit_ids) != len(fit) or len(query_ids) != len(query) or fit_ids & query_ids:
        raise ValueError("fit and query IDs are duplicated or overlap")
    fit_groups = {r[group_key] for r in fit}
    query_groups = {r[group_key] for r in query}
    if fit_groups & query_groups:
        raise ValueError(f"fit and query {group_key} values overlap")


def read_ids(path: Path, key: str = "sample_id") -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "sample_ids" in payload:
            return [str(x) for x in payload["sample_ids"]]
        if isinstance(payload, list):
            return [str(x[key] if isinstance(x, dict) else x) for x in payload]
        raise ValueError(f"unsupported ID JSON: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return [r[key] for r in csv.DictReader(stream)]
