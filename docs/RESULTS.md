# Saved prediction provenance

`records/predictions/` contains 14 small `.npz` archives with sample IDs and 11-class probabilities only. Each archive was derived from a frozen historical prediction artifact; `records/prediction_index.json` records SHA-256 of both exported files and their source artifacts. The source artifacts themselves were not uploaded. `records/historical_metrics.json` was recomputed from these vectors and the published manifest, not transcribed from a summary table. Run `PYTHONPATH=src python scripts/validate_release.py` to check every archive hash and score.

Full Test fusion and ConvNeXt each contain 1,742 rows. Development contains K=5/10/20, three draws each, 220 rows per draw. K=20 subset Test contains three draws, 550 rows each. The 11-class macro-F1 and balanced accuracy are computed without hiding missing classes. Full precision is retained in JSON; README displays percentage values rounded to three decimals.

The 2026-11-06 historical result binding file is included as `historical_result_bindings.json` solely to identify older source artifacts. It does not establish the later source-audited cohort. Comparisons without published prediction vectors are reported only as manuscript numbers and marked accordingly.
