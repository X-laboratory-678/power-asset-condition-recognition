# Saved prediction provenance

## Historical results and audit status

The saved historical full Test predictions contain 1,742 images. The original Test split was accessed during development, so these values describe that historical experiment. The manuscript's later score-blind source audit reports 1,492 retained full Test images and a ranking in which ConvNeXt-Tiny is above the proposed fusion. Its accepted near-sequence links and unresolved-component IDs are absent from the available records; exact audited scores are therefore **unverified in this release**. See [audit inputs and rule](AUDIT.md).

| Full-data path | Historical Test Accuracy / Macro-F1 / BA (%) | Evidence |
| --- | ---: | --- |
| Joint objective + visual fusion | 98.507 / 97.570 / 96.961 | Recomputed from published predictions |
| ConvNeXt-Tiny | 98.278 / 97.293 / 96.721 | Recomputed from published predictions |
| YOLOv8n-cls | 95.695 / 92.682 / 91.941 | Manuscript-reported; prediction vectors unavailable |
| HOG + RBF-SVM | 81.745 / 61.473 / 61.949 | Manuscript-reported; prediction vectors unavailable |

For limited-label Development (K=5, 10, and 20 images per class), the published historical fusion predictions give three-draw mean accuracies of **76.212%, 87.424%, and 92.727%**. The K=20 historical 550-image Test subset gives 88.545% Accuracy and 88.429% Macro-F1. The manuscript reports later retained cohorts of 209 Development and 525 subset Test images; their exact row lists and audited scores are unverified in the available records.

## Prediction files

`records/predictions/` contains 14 indexed historical `.npz` archives with sample IDs and 11-class probabilities. Each was derived from a frozen historical prediction artifact; `records/prediction_index.json` records SHA-256 of both exported files and their source artifacts. `records/historical_metrics.json` was recomputed from these vectors and the published manifest. Run `PYTHONPATH=src python scripts/validate_release.py` to check every historical archive hash and score. Functional-run predictions in the same directory are indexed separately by `records/validation_matrix.json` and checked with `scripts/verify_validation_matrix.py`.

Full Test fusion and ConvNeXt each contain 1,742 rows. Development contains K=5/10/20, three draws each, 220 rows per draw. K=20 subset Test contains three draws, 550 rows each. Macro-F1 uses the fixed 11-class label set, including zero-recall classes. JSON retains full precision; displayed percentages are rounded to three decimals.

The 2026-11-06 historical result binding file identifies older source artifacts. The later source-audited cohort requires the decisions described in [the audit guide](AUDIT.md). Comparisons without published prediction vectors retain their manuscript attribution. Small-scale functional checks and their hashes are recorded separately in [`validation_matrix.json`](../records/validation_matrix.json).
