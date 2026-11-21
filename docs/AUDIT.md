# Source-isolation audit status

The manuscript's main results use historical image-group splits. It later reports a score-blind source audit based on exact file hashes, spatial/acquisition links and accepted near-sequence relationships. The manuscript gives retained counts of **1,492** full Test, **209** Development, and **525** subset Test, and says ConvNeXt-Tiny ranks above the proposed fusion on retained full Test. Historical Test images were accessed during development. Source components also cross some original full-data OOF boundaries, while some support-gate fold relationships remain unresolved.

The available local records include stable IDs, file hashes, historical group IDs, source IDs, supports, query sets and prediction vectors. They **do not include the later audit's accepted near-sequence edge list or unresolved-component IDs**. Hash/source/group unions alone retain all historical queries; thus they do not reconstruct the manuscript's 1,492/209/525 cohorts. No retained metric is published here as independently recomputed, and the ranking statement is attributed to the manuscript only.

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

The count assertion prevents silently substituting a different cohort. A later release should add the actual accepted link and unresolved decisions, their review provenance, and separately recomputed retained metrics. A genuinely unused source-independent cohort would still be needed for prospective confirmation.
