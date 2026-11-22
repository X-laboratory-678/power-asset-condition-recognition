# Experiment entry points

Use the fixed manifest, class order and `configs/paper.json`. Set `DATA_ROOT` to the downloaded `defect_supervised` tree and model variables to local model directories or model hub IDs. Each public command writes to an ignored output directory. Model weights and large features stay local.

Full-data: `train-full` fits one recorded member from scratch; run `member0`, `member1`, `member2` independently and call `fuse-full`. `--objective response`, `poolce` and `joint` expose the response-loss and PoolCE ablations in **Response–Representation Coupled Shaping**. Query labels are removed before model workers. PoolCE's image-state forward receives only the input prefix. The exported task readout is refit on each member's training features; **Native Visual Reference Fusion** uses the matching member's frozen visual features and recorded fusion pools and weights.

Limited-label training (K-shot with K=5, 10, or 20 support images per class): `generate-teacher` creates eight ordered observation fields for current supports only and resumes a matching partial cache; `train-fewshot` checks teacher coverage, starts a new Qwen3-VL adapter, trains a direct head, extracts frozen Qwen2-VL and DINOv2 features, selects the partner classifier by support-only group CV, and fuses. Qwen3 adaptation and DINOv2 use the first visible GPU; frozen Qwen2-VL uses Hugging Face automatic placement across visible GPUs. K=5 uses fixed raw 0.25/0.75 fusion. K=10/20 fits branch temperatures and entropy coefficient from the recorded two-fold support OOF assignments. `fused.npz` retains both raw branches and fixed control probabilities. For a small dynamic-gate check at K=10/20, keep the full support membership and use `--smoke-queries-per-class 1 --max-updates 1`; `--smoke-per-class` switches to fixed fusion. Smoke options validate execution, while paper metrics use the full update budget.

Generate each support cache before training its K=10/20 model:

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

Representative baselines:

```bash
pacr baseline --method hog --data-root "$DATA_ROOT" --seed 20261301 --output outputs/hog
pacr baseline --method convnext --data-root "$DATA_ROOT" --seed 20261301 --output outputs/convnext
pacr baseline --method yolo --yolo-checkpoint yolov8n-cls.pt \
  --data-root "$DATA_ROOT" --seed 20261301 --output outputs/yolo
pacr frozen-qwen3-ridge --data-root "$DATA_ROOT" --qwen3-model "$QWEN3" \
  --seed 20261301 --output outputs/qwen3_ridge.npz
```

These commands train fresh representative comparisons with the published configuration. The paper's historical comparison numbers came from separate training runs and are attributed in [the result record](RESULTS.md). Use Development for configuration selection; `full_test` names the historical Test split. Score predictions with `pacr score --predictions PATH`, optionally restricted to a frozen retained-ID CSV. LBP, SIFT–BoVW, ResNet-50, YOLO–QwenVL, iSAFE, DefectGLM and SAVs are manuscript-reported comparisons rather than executable paths in this release.
