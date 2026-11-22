# Data reconstruction

The release provides path-free manifests, image hashes, class labels and membership tables. Obtain InsPLAD from the [authors' page](https://andreluizbvs.github.io/InsPLAD/) or [Mendeley Data](https://data.mendeley.com/datasets/5n3fjgvfyz/1). The authors' page displays **CC BY-NC-SA 4.0** and the Mendeley record displays **CC BY-NC 3.0**; check the applicable upstream terms before redistributing images. Apache-2.0 covers original code in this repository.

Extract the defect-supervised tree as `DATA_ROOT`. Every row in `data/manifests/insplad_clean_v1.csv` points to a relative file under that tree. The manifest has SHA-256 for each file, a stable sample ID, 11-way leaf label, original split, historical group ID, and source ID. Reconstruct and check it with:

```bash
pacr validate-data --data-root "$DATA_ROOT" --verify-images
```

Manifest SHA-256: `6d58a954fffa832e5787e87625ba7133397140db616b3f5776092fab376fb537`. Expected historical counts are 8,039 train, 1,744 val, 1,742 test. The manifest specifies the research workspace's `INSPLAD-CLEAN-V1` selection. Archive variants may differ in paths or bytes; match the archive version and extraction layout before training.

| ID | Leaf class |
| ---: | --- |
| 0 | `glass-insulator::good` |
| 1 | `glass-insulator::missing-cap` |
| 2 | `lightning-rod-suspension::good` |
| 3 | `lightning-rod-suspension::rust` |
| 4 | `polymer-insulator-upper-shackle::good` |
| 5 | `polymer-insulator-upper-shackle::rust` |
| 6 | `vari-grip::bird-nest` |
| 7 | `vari-grip::good` |
| 8 | `vari-grip::rust` |
| 9 | `yoke-suspension::good` |
| 10 | `yoke-suspension::rust` |

The fixed full-data member IDs are in `data/provenance/full_members.csv`; the original outer fold IDs are in `historical_outer_folds.csv`. For limited-label runs, `fewshot_supports.csv` contains all three support draws for K=5/10/20, `fewshot_queries.csv` lists Development and subset-Test queries, and `fewshot_oof_folds.csv` contains the K=10/20 two-fold support-only assignments. `scripts/validate_release.py` checks class counts, train/query group isolation, OOF fold coverage and boundaries, prediction hashes, and saved metric agreement. Historical group IDs predate the later source audit; the [audit guide](AUDIT.md) records the source-link methodology.

`scripts/export_local_metadata.py` is an archival export tool for the original author's research workspace. Users of this release need only the committed path-free files and the upstream data archive.
