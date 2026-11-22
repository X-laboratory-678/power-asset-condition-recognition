# Response-Trained Representations and Visual Reference Fusion for Power-Asset Condition Recognition

This repository implements **Response–Representation Coupled Shaping** and **Native Visual Reference Fusion** for 11 equipment–condition classes in InsPLAD, across full-data and limited-label settings.

## Method

**Response–Representation Coupled Shaping** trains shared Qwen3-VL adapters with class-response supervision and **PoolCE on pre-answer image states**. **Native Visual Reference Fusion** combines the adapted image-state readout with a readout of frozen native visual features in each of three full-data members. In the limited-label setting (K=5, 10, or 20 images per class), eight-field support observations and class labels train a direct head, while a frozen Qwen2-VL–DINOv2 branch supplies the visual reference. K=5 uses fixed fusion; K=10/20 fit an entropy gate from support-only out-of-fold predictions.

| Path | Entry point | Published protocol |
| --- | --- | --- |
| Data and membership checks | `pacr validate-data` | 11,525 image hashes, 11 classes, recorded groups and support folds |
| Full-data adaptation and fusion | `pacr train-full`, `pacr fuse-full` | Three members; response, PoolCE and joint objectives; Native Visual Reference Fusion |
| Limited-label adaptation and fusion | `pacr generate-teacher`, `pacr train-fewshot` | Three draws per K; direct head, frozen visual reference, fixed or support OOF fusion |
| Representative comparisons | `pacr baseline`, `pacr frozen-qwen3-ridge` | ConvNeXt-Tiny, YOLOv8n-cls, HOG-SVM, frozen Qwen3 + Ridge |
| Evaluation and source audit | `pacr score`, `pacr audit-cohort` | Saved probabilities, metric recomputation and source-component filtering |

The published historical predictions re-score to **98.507% accuracy** on the 1,742-image full Test and **92.727% mean accuracy** across three K=20 Development draws. The [experiment guide](docs/EXPERIMENTS.md) gives baseline and ablation commands; the [result record](docs/RESULTS.md) documents the historical predictions, manuscript comparisons, and source-audited cohort. The historical Test was part of development; the result record describes the retained-cohort evidence and reconstruction inputs.

## Get started

Use Python 3.10 and a compatible CUDA PyTorch build for Qwen training. Package versions are pinned in `pyproject.toml`. Install with `pip install -e .`. Obtain the InsPLAD archive and model weights separately under their upstream terms. `DATA_ROOT` is the extracted `defect_supervised` directory; the [data guide](docs/DATA.md) documents its 11-class mapping and hash checks.

```bash
pacr validate-data --data-root "$DATA_ROOT" --verify-images
PYTHONPATH=src python scripts/validate_release.py
PYTHONPATH=src python scripts/recompute_historical.py
PYTHONPATH=src python scripts/verify_validation_matrix.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run the full-data ensemble:

```bash
for member in member0 member1 member2; do
  pacr train-full --data-root "$DATA_ROOT" --cohort development --member "$member" \
    --qwen3-model "$QWEN3" --seed 20261301 --output "outputs/full/$member"
done
pacr fuse-full --members outputs/full/member0 outputs/full/member1 outputs/full/member2 \
  --output outputs/full/fused.npz
pacr score --predictions outputs/full/fused.npz
```

Run one K=5 support draw:

```bash
pacr generate-teacher --data-root "$DATA_ROOT" --cohort development --draw 0 --k 5 \
  --qwen3-model "$QWEN3" --output outputs/teacher_draw0_k5.jsonl
pacr train-fewshot --data-root "$DATA_ROOT" --cohort development --draw 0 --k 5 \
  --qwen3-model "$QWEN3" --qwen2-model "$QWEN2" --dino-model "$DINO" \
  --teacher-cache outputs/teacher_draw0_k5.jsonl --seed 20261301 --output outputs/fewshot_draw0_k5
pacr score --predictions outputs/fewshot_draw0_k5/fused.npz
```

Run K=10 and K=20 with their own support teacher caches and the recorded two-fold support OOF gate:

```bash
for k in 10 20; do
  pacr generate-teacher --data-root "$DATA_ROOT" --cohort development --draw 0 --k "$k" \
    --qwen3-model "$QWEN3" --output "outputs/teacher_draw0_k${k}.jsonl"
  pacr train-fewshot --data-root "$DATA_ROOT" --cohort development --draw 0 --k "$k" \
    --qwen3-model "$QWEN3" --qwen2-model "$QWEN2" --dino-model "$DINO" \
    --teacher-cache "outputs/teacher_draw0_k${k}.jsonl" --seed 20261301 \
    --output "outputs/fewshot_draw0_k${k}"
  pacr score --predictions "outputs/fewshot_draw0_k${k}/fused.npz"
done
```

Model references accept a local directory or model hub ID. Qwen3 adaptation and DINOv2 use the first visible GPU; frozen Qwen2-VL uses Hugging Face automatic placement across visible GPUs. Outputs stay in Git-ignored directories. The [validation matrix](records/validation_matrix.json) records small-scale end-to-end checks and their lightweight predictions separately from paper-scale results.

## Data and license

Download InsPLAD from the [authors' page](https://andreluizbvs.github.io/InsPLAD/) or [Mendeley Data](https://data.mendeley.com/datasets/5n3fjgvfyz/1). The two pages display different license identifiers (CC BY-NC-SA 4.0 and CC BY-NC 3.0); check the applicable upstream terms. This repository distributes path-free metadata and small predictions. The [Apache-2.0 license](LICENSE) covers original repository code, while images and upstream models retain their own terms.
