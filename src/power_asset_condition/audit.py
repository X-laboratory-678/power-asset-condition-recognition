"""Score-blind source-component audit for a supplied, frozen link decision set.

Near-sequence links require an explicit accepted-links file. The tool does not
infer or silently accept new links from evaluation labels or predictions.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self, ids):
        self.parent = {sid: sid for sid in ids}

    def find(self, sid):
        parent = self.parent[sid]
        if parent != sid:
            self.parent[sid] = self.find(parent)
        return self.parent[sid]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def source_components(rows: list[dict], accepted_links: Path | None = None) -> dict[str, str]:
    ids = {r["sample_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate sample ID in source audit")
    uf = UnionFind(ids)
    for key in ("group_id", "source_id", "sha256"):
        seen = {}
        for row in rows:
            value = row.get(key)
            if value:
                if value in seen:
                    uf.union(row["sample_id"], seen[value])
                else:
                    seen[value] = row["sample_id"]
    if accepted_links is not None:
        for link in _csv(accepted_links):
            left, right = link["sample_id_a"], link["sample_id_b"]
            if left not in ids or right not in ids or left == right:
                raise ValueError("near-sequence link has missing or identical endpoint")
            if link.get("decision", "accepted") != "accepted":
                continue
            uf.union(left, right)
    return {sid: uf.find(sid) for sid in ids}


def retained_query_ids(rows: list[dict], components: dict[str, str], query_ids: set[str],
                       fit_or_selection_ids: set[str], unresolved_ids: set[str]) -> tuple[set[str], dict]:
    ids = {r["sample_id"] for r in rows}
    if not (query_ids | fit_or_selection_ids | unresolved_ids).issubset(ids):
        raise ValueError("audit references unknown sample IDs")
    blocked = {components[sid] for sid in fit_or_selection_ids | unresolved_ids}
    retained = {sid for sid in query_ids if components[sid] not in blocked}
    group_members = defaultdict(set)
    for sid in query_ids:
        group_members[components[sid]].add(sid)
    return retained, {
        "query_rows": len(query_ids), "retained_rows": len(retained),
        "excluded_rows": len(query_ids) - len(retained),
        "query_components": len(group_members),
        "retained_components": len({components[sid] for sid in retained}),
    }


def write_retained(path: Path, ids: set[str], components: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "audit_component_id"])
        writer.writeheader()
        for sid in sorted(ids):
            writer.writerow({"sample_id": sid, "audit_component_id": components[sid]})
