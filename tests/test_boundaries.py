"""Small adversarial checks for score-blind audit and fusion boundaries."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from power_asset_condition.audit import retained_query_ids, source_components
from power_asset_condition.fusion import EntropyGate, fixed_fewshot, full_ensemble


class BoundaryTests(unittest.TestCase):
  def test_accepted_link_and_unresolved_component_exclusion(self):
    rows = [
        {"sample_id": "fit", "group_id": "g1", "source_id": "s1", "sha256": "a"},
        {"sample_id": "near_query", "group_id": "g2", "source_id": "s2", "sha256": "b"},
        {"sample_id": "clean_query", "group_id": "g3", "source_id": "s3", "sha256": "c"},
        {"sample_id": "unresolved_query", "group_id": "g4", "source_id": "s4", "sha256": "d"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        links = Path(directory) / "accepted.csv"
        with links.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id_a", "sample_id_b", "decision"])
            writer.writeheader()
            writer.writerow({"sample_id_a": "fit", "sample_id_b": "near_query", "decision": "accepted"})
        components = source_components(rows, links)
    retained, report = retained_query_ids(
        rows, components, {"near_query", "clean_query", "unresolved_query"},
        {"fit"}, {"unresolved_query"})
    self.assertEqual(retained, {"clean_query"})
    self.assertEqual(report["excluded_rows"], 2)


  def test_fusion_probabilities_and_support_only_gate(self):
    task = np.full((11, 11), 0.01)
    reference = np.full((11, 11), 0.01)
    for i in range(11):
        task[i, i] = 0.9
        reference[i, (i + 1) % 11] = 0.9
    task /= task.sum(axis=1, keepdims=True)
    reference /= reference.sum(axis=1, keepdims=True)
    fixed = fixed_fewshot(task, reference)
    np.testing.assert_allclose(fixed.sum(axis=1), 1.0)
    gate = EntropyGate.fit(task, reference, np.arange(11))
    dynamic = gate.predict(task, reference)
    np.testing.assert_allclose(dynamic.sum(axis=1), 1.0)
    self.assertTrue(np.isfinite(dynamic).all())
    np.testing.assert_allclose(
        full_ensemble(np.stack([task] * 3), np.stack([reference] * 3)).sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
