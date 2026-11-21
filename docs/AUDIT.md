# Source-isolation audit status

The manuscript's main results use historical image-group splits, and historical Test images were accessed during development. A later score-blind source audit uses exact hashes, spatial/acquisition links and accepted near-sequence relationships. The manuscript reports 1,492 retained full Test, 209 retained Development and 525 retained subset-Test images, with ConvNeXt-Tiny above the proposed fusion on retained full Test. Source components cross some original full-data OOF boundaries, and some support-gate fold relationships remain unresolved. The release preserves stable IDs, hashes, historical group and source IDs, support/query memberships, and prediction vectors. Exact audited scores require the accepted near-sequence links and unresolved-component IDs, which are not included here; `pacr audit-cohort` applies the frozen rule when those inputs are supplied.

`pacr audit-cohort` implements a frozen, score-blind rule: union rows sharing historical group ID, source ID or exact SHA-256; add **only** links with an accepted decision in an explicit supplied CSV; exclude entire components that touch fit/selection IDs or the supplied unresolved IDs; emit the retained query ID list and a count report. Freeze both input files before reading prediction outcomes. Example input headers:

```text
accepted_links.csv: sample_id_a,sample_id_b,decision
unresolved.csv: sample_id
```

Example:

```bash
pacr audit-cohort --accepted-links accepted_links.csv --unresolved unresolved.csv \
  --queries query_ids.csv --fit-or-selection fit_selection_ids.csv \
  --output outputs/retained.csv
pacr score --predictions records/predictions/full_test_fusion.npz \
  --retained-ids outputs/retained.csv --expected-rows 1492
```

The count assertion fixes the intended cohort size. Exact retained metrics can be recomputed when the accepted-link decisions and unresolved IDs are supplied with their review provenance.
