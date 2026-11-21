# Power-asset condition recognition

Code and small provenance records for **Response-Trained Representations and Visual Reference Fusion for Power-Asset Condition Recognition** (`main.pdf`, the author's manuscript). The task is 11-way InsPLAD equipment–condition classification under full-data and K=5/10/20 support budgets.

## What is here

| Component | Public entry point | Scope |
| --- | --- | --- |
| Manifest and upstream archive reconstruction | `pacr validate-data` | 11,525 image IDs, SHA-256, relative paths, class and historical split |
| Full-data Qwen3-VL response + pre-answer PoolCE | `pacr train-full` | Three recorded training memberships; objective controls `response`, `poolce`, `joint` |
| Frozen same-model visual readout and fusion | `pacr fuse-full` | Mean/mean/mean–max visual pools, weights 0.9/0.7/0.7, member averaging |
| Few-shot structured response and direct head | `pacr generate-teacher`, `pacr train-fewshot` | Three recorded support draws at each K; K=5 fixed fusion, K=10/20 support OOF entropy gate |
| Frozen comparison paths | `pacr frozen-qwen3-ridge`, `pacr baseline` | Qwen3 Ridge; ConvNeXt-Tiny, YOLOv8n-cls, HOG-SVM |
| Metric and source audit tools | `pacr score`, `pacr audit-cohort` | Score saved label-free probabilities; form source components from a supplied decision file |

Other methods in the manuscript are **historical reported comparisons**, not runnable baselines in this release. The response/PoolCE and raw/fixed/dynamic fusion switches are available for ablation; this repository does not claim that rerunning them reproduces every archived training environment or score. No image, pretrained model, checkpoint, or feature cache is distributed.

## Historical results and source audit

The original historical Test split was accessed during development. These numbers are descriptive historical results, **not** a fresh untouched Test evaluation. `records/historical_metrics.json` recomputes the available values from the committed lightweight prediction archives. The manuscript reports a later score-blind source audit; its accepted near-sequence edge decisions and unresolved-component ID list were not found in the available research workspace, so its exact retained cohorts and scores cannot be regenerated from this release. Do not treat the historical Test values as source-independent estimates.

| Full-data path | Historical 1,742 Test Accuracy / Macro-F1 / BA (%) | Later source-audited Test |
| --- | ---: | --- |
| Proposed joint objective + visual fusion | 98.507 / 97.570 / 96.961 (saved predictions rechecked) | Manuscript states 1,492 retained and a changed ranking; exact value **unverified here** |
| ConvNeXt-Tiny | 98.278 / 97.293 / 96.721 (saved predictions rechecked) | Manuscript states ConvNeXt ranks above proposed; exact value **unverified here** |
| YOLOv8n-cls | 95.695 / 92.682 / 91.941 (manuscript only) | **Unverified here** |
| HOG + RBF-SVM | 81.745 / 61.473 / 61.949 (manuscript only) | **Unverified here** |

For few-shot, the saved historical Development fusion accuracies (three-draw means, K=5/10/20) are 76.212/87.424/92.727%. The saved K=20 historical 550-image Test subset gives 88.545% Accuracy and 88.429% Macro-F1 (three-draw means). The manuscript describes later retained counts of 209 Development and 525 subset Test images; their exact row lists and scores are **unverified here**. See [audit status](docs/AUDIT.md) and [prediction provenance](docs/RESULTS.md).

## Setup and data

Python 3.10 and a CUDA GPU are required for Qwen training. The pinned environment is in `pyproject.toml`; install a compatible CUDA PyTorch build for your system before `pip install -e .`. CPU is sufficient for manifest checks, metric recomputation and HOG-SVM. You must separately obtain InsPLAD, Qwen3-VL-8B-Instruct, Qwen2-VL-7B-Instruct, DINOv2-B/14, and baseline pretrained weights under their own terms.

Download the defect-supervised InsPLAD archive from the [authors' page](https://andreluizbvs.github.io/InsPLAD/) or [Mendeley Data](https://data.mendeley.com/datasets/5n3fjgvfyz/1), then extract it so `DATA_ROOT/glass-insulator/train/good/...` exists. Use [data instructions](docs/DATA.md) to verify the archive and class mapping. The authors' page identifies CC BY-NC-SA 4.0, while Mendeley identifies CC BY-NC 3.0; verify the applicable upstream terms yourself. Apache-2.0 applies only to original code in this repository and does not license the images or upstream models.

```bash
pacr validate-data --data-root "$DATA_ROOT" --verify-images
PYTHONPATH=src python scripts/validate_release.py
PYTHONPATH=src python scripts/recompute_historical.py
```

## Run the methods

All model references accept a local directory or a model hub ID. `--data-root` is the extracted `defect_supervised` directory. Outputs are ignored by Git and may contain large model artifacts.

```bash
for member in member0 member1 member2; do
  pacr train-full --data-root "$DATA_ROOT" --member "$member" \
    --qwen3-model "$QWEN3" --output "outputs/full/$member"
done
pacr fuse-full --members outputs/full/member0 outputs/full/member1 outputs/full/member2 \
  --output outputs/full/fused.npz
pacr score --predictions outputs/full/fused.npz
```

```bash
pacr generate-teacher --data-root "$DATA_ROOT" --draw 0 --k 5 \
  --qwen3-model "$QWEN3" --output outputs/teacher_draw0_k5.jsonl
pacr train-fewshot --data-root "$DATA_ROOT" --draw 0 --k 5 \
  --qwen3-model "$QWEN3" --qwen2-model "$QWEN2" --dino-model "$DINO" \
  --teacher-cache outputs/teacher_draw0_k5.jsonl --output outputs/fewshot_draw0_k5
pacr score --predictions outputs/fewshot_draw0_k5/fused.npz
```

Set `--k 10` or `--k 20` for the support OOF entropy gate. The CLI uses the recorded original fold memberships in `data/provenance/fewshot_oof_folds.csv`. For a short functional run, add `--smoke-per-class 2 --smoke-queries-per-class 1 --max-updates 1` at K=5; such a run does not reproduce paper metrics. Baseline examples and method constraints are in [experiments](docs/EXPERIMENTS.md).

## License

Original repository code: Apache-2.0, see [LICENSE](LICENSE). Dataset, pretrained models and copied third-party components retain their upstream terms.
