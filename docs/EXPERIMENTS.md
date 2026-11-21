# Experiment entry points

Use the fixed manifest, class order and `configs/paper.json`. Set `DATA_ROOT` to the downloaded `defect_supervised` tree and model variables to local model directories or model hub IDs. Each public command writes to an ignored output directory. Model weights and large features stay local.

Full data: `train-full` fits one recorded member from scratch; run `member0`, `member1`, `member2` independently and call `fuse-full`. `--objective response`, `poolce` and `joint` expose the response-loss and PoolCE ablations. Query labels are removed before model workers. PoolCE's image-state forward receives only the input prefix. The exported task readout is refit on each member's training features; the frozen visual reference readout uses the matching member's visual features. `fuse-full` uses the recorded pools and weights.

Few-shot: `generate-teacher` creates eight ordered observation fields for current supports only; `train-fewshot` checks teacher coverage, starts a new Qwen3-VL adapter, trains a direct head, extracts frozen Qwen2-VL and DINOv2 features, selects the partner classifier by support-only group CV, and fuses. K=5 uses fixed raw 0.25/0.75 fusion. K=10/20 fits branch temperatures and entropy coefficient from the recorded two-fold support OOF assignments. `fused.npz` retains both raw branches and fixed control probabilities. `--max-updates` and smoke subset options are functional checks and must not be used for result claims.

Representative baselines:

```bash
pacr baseline --method hog --data-root "$DATA_ROOT" --output outputs/hog
pacr baseline --method convnext --data-root "$DATA_ROOT" --output outputs/convnext
pacr baseline --method yolo --yolo-checkpoint yolov8n-cls.pt \
  --data-root "$DATA_ROOT" --output outputs/yolo
pacr frozen-qwen3-ridge --data-root "$DATA_ROOT" --qwen3-model "$QWEN3" \
  --output outputs/qwen3_ridge.npz
```

The baseline entry points are runnable comparison implementations. They are not asserted bit-for-bit recreations of historical runs: those used separate training setups and model checkpoints. The `full_test` cohort is the historical split; avoid using it for configuration selection. Metrics can be computed with `pacr score --predictions PATH`, optionally restricted to a frozen retained ID CSV. This release does not include runnable LBP, SIFT–BoVW, ResNet-50, YOLO–QwenVL, iSAFE, DefectGLM or SAVs baselines; their manuscript values remain historical references only.
