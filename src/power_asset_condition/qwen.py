"""Qwen3-VL response and pre-answer image-state adaptation.

The response loss and PoolCE use separate forwards. PoolCE always receives
only the user/image prefix, so assistant target tokens cannot enter its image
representation. The visual tower, merger, base language model, and LM head
remain frozen; only language-layer LoRA and the PoolCE head are optimized.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from . import OBSERVATION_KEYS
from .readout import fit_logistic, predict_logistic

LORA_TARGETS = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


def _seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base(model_ref: str, config: dict):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL training needs one CUDA GPU")
    processor = AutoProcessor.from_pretrained(model_ref)
    if not hasattr(processor.image_processor, "max_pixels"):
        raise RuntimeError("Qwen3-VL processor has no max_pixels field")
    processor.image_processor.max_pixels = int(config["image_max_pixels"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=["model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_ref, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
    for param in model.parameters():
        param.requires_grad_(False)
    model.config.use_cache = False
    return model, processor


def attach_language_lora(base, config: dict, seed: int):
    import torch
    from peft import LoraConfig, get_peft_model

    _seed(seed, torch)
    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]), bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    trainable = []
    for name, param in model.named_parameters():
        active = ("lora_A" in name or "lora_B" in name) and "language_model.layers." in name
        param.requires_grad_(active)
        if active:
            trainable.append(param)
        if active and ("visual" in name or "lm_head" in name):
            raise RuntimeError("visual/LM head unexpectedly trainable")
    if not trainable:
        raise RuntimeError("language-layer LoRA target match is empty")
    return model, trainable


def _batch(processor, row: dict, prompt: str, target: str | None, config: dict):
    import torch

    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    def user():
        return {"role": "user", "content": [
            {"type": "image", "image": image.copy()}, {"type": "text", "text": prompt}]}
    prefix = processor.apply_chat_template([user()], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    if target is None:
        packed = prefix
        prefix_length = int(prefix["input_ids"].shape[1])
    else:
        packed = processor.apply_chat_template(
            [user(), {"role": "assistant", "content": [{"type": "text", "text": target}]}],
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
        a, b = prefix["input_ids"][0].tolist(), packed["input_ids"][0].tolist()
        prefix_length = len(a)
        if b[:prefix_length] != a or len(b) <= prefix_length:
            raise RuntimeError("assistant response boundary does not match prefix")
        if len(b) > int(config["sequence_cutoff"]) or len(b) - prefix_length > int(config["assistant_target_token_cap"]):
            raise RuntimeError("training example exceeds registered token limits")
        labels = packed["input_ids"].clone()
        labels[:, :prefix_length] = -100
        packed["labels"] = labels
    moved = {key: (value.to("cuda:0", dtype=torch.bfloat16) if value.is_floating_point()
                   else value.to("cuda:0")) for key, value in packed.items() if hasattr(value, "to")}
    return moved, prefix_length


def _model_inputs(batch: dict, prefix_length: int | None = None) -> dict:
    sequence_length = int(batch["input_ids"].shape[-1])
    excluded = {"labels"}
    non_sequence = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    out = {}
    for key, value in batch.items():
        if key in excluded:
            continue
        if prefix_length is not None and key not in non_sequence and value.ndim >= 1 and value.shape[-1] == sequence_length:
            out[key] = value[..., :prefix_length]
        else:
            out[key] = value
    return out


def _image_states(model, batch: dict, *, capture_visual: bool = False):
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = base.model.visual
    captured = []
    hook = None
    if capture_visual:
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else getattr(output, "pooler_output", output)
            if isinstance(value, (tuple, list)):
                value = value[0]
            captured.append(value)
        hook = visual.register_forward_hook(capture)
    try:
        hidden = base.model(**_model_inputs(batch), use_cache=False).last_hidden_state
    finally:
        if hook is not None:
            hook.remove()
    mask = batch["input_ids"][0].eq(int(model.config.image_token_id))
    if not bool(mask.any()):
        raise RuntimeError("input has no image tokens")
    lm_tokens = hidden[0, mask, :].float()
    if lm_tokens.shape[1] != 4096 or not torch.isfinite(lm_tokens).all():
        raise RuntimeError("invalid 4096-wide Qwen3 language image states")
    if capture_visual:
        if len(captured) != 1 or captured[0].ndim != 2 or captured[0].shape[1] != 4096:
            raise RuntimeError("invalid frozen visual-merger activation")
        return lm_tokens, captured[0].float()
    return lm_tokens, None


def extract_features(model, processor, rows: list[dict], prompt: str, config: dict, visual: bool = True) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    lm_mean, visual_mean, visual_max = [], [], []
    with torch.inference_mode():
        for row in rows:
            batch, _ = _batch(processor, row, prompt, None, config)
            lm, vis = _image_states(model, batch, capture_visual=visual)
            lm_mean.append(lm.mean(0).cpu().numpy())
            if visual:
                visual_mean.append(vis.mean(0).cpu().numpy())
                visual_max.append(vis.max(0).values.cpu().numpy())
            del batch, lm, vis
    result = {"lm_mean": np.stack(lm_mean).astype(np.float32)}
    if visual:
        result["visual_mean"] = np.stack(visual_mean).astype(np.float32)
        result["visual_max"] = np.stack(visual_max).astype(np.float32)
    return result


def _head_from_features(base_features: np.ndarray, labels: np.ndarray):
    import torch

    scaler, classifier = fit_logistic(base_features, labels, c=1.0)
    return {"mean": torch.tensor(scaler.mean_, device="cuda:0", dtype=torch.float32),
            "scale": torch.tensor(scaler.scale_, device="cuda:0", dtype=torch.float32),
            "weight": torch.nn.Parameter(torch.tensor(classifier.coef_, device="cuda:0", dtype=torch.float32)),
            "bias": torch.nn.Parameter(torch.tensor(classifier.intercept_, device="cuda:0", dtype=torch.float32))}


def _losses(model, batch: dict, prefix_length: int, head: dict, label_id: int, objective: str):
    import torch
    import torch.nn.functional as F

    base = model.get_base_model()
    sft = torch.zeros((), device="cuda:0")
    pool = torch.zeros((), device="cuda:0")
    if objective in ("response", "joint"):
        hidden = base.model(**_model_inputs(batch), use_cache=False).last_hidden_state
        logits = base.lm_head(hidden).float()
        shifted = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                  batch["labels"][:, 1:].reshape(-1),
                                  ignore_index=-100, reduction="none")
        valid = batch["labels"][:, 1:].ne(-100).reshape(-1)
        if not bool(valid.any()):
            raise RuntimeError("no supervised response tokens")
        sft = shifted[valid].mean()
    if objective in ("poolce", "joint"):
        prefix = _model_inputs(batch, prefix_length)
        lm, _ = _image_states(model, prefix, capture_visual=False)
        pooled = lm.mean(0)
        logits = F.linear((pooled - head["mean"]) / head["scale"], head["weight"], head["bias"])
        pool = F.cross_entropy(logits[None], torch.tensor([label_id], device="cuda:0"))
    return sft, pool


def train(model_ref: str, config: dict, fit_rows: list[dict], query_rows: list[dict],
          output_dir: Path, *, mode: str, objective: str = "joint", teacher: dict | None = None,
          seed: int = 20261301, max_updates: int | None = None) -> dict:
    """Train one independent member/draw and write its label-free predictions.

    `mode=full` uses one epoch/16 accumulation; `mode=fewshot` uses the
    registered 32/64/128 update budget and four-way accumulation. `max_updates`
    is a smoke-test override and must not be used for reported paper scores.
    """
    import bitsandbytes as bnb
    import torch

    if mode not in ("full", "fewshot") or objective not in ("response", "poolce", "joint"):
        raise ValueError("invalid training mode/objective")
    if any("leaf_id" in row or "label_id" in row for row in query_rows):
        raise ValueError("query workers must receive label-free rows")
    labels = np.asarray([row["leaf_id"] for row in fit_rows], dtype=np.int64)
    if set(labels.tolist()) != set(range(11)):
        raise ValueError("fit rows must cover all 11 classes")
    prompt = config["full_prompt" if mode == "full" else "fewshot_prompt"]
    if mode == "fewshot" and teacher is None:
        raise ValueError("structured few-shot training requires support-only teacher observations")
    _seed(seed, torch)
    base, processor = load_base(model_ref, config)
    base_features = extract_features(base, processor, fit_rows, prompt, config, visual=False)["lm_mean"]
    head = _head_from_features(base_features, labels)
    model, lora_params = attach_language_lora(base, config, seed)
    trainable = list(lora_params)
    if objective in ("poolce", "joint"):
        trainable += [head["weight"], head["bias"]]
    optimizer = bnb.optim.AdamW8bit([
        {"params": lora_params, "lr": config["lora_lr"], "weight_decay": config["weight_decay"]},
        {"params": [head["weight"], head["bias"]], "lr": config["poolce_lr"], "weight_decay": config["weight_decay"]},
    ] if objective != "response" else [{"params": lora_params, "lr": config["lora_lr"]}])
    accumulation = int(config["full_gradient_accumulation" if mode == "full" else "fewshot_gradient_accumulation"])
    if mode == "full":
        updates = math.ceil(len(fit_rows) / accumulation)
    elif len(fit_rows) // 11 in (5, 10, 20) and len(fit_rows) % 11 == 0:
        updates = int(config["fewshot_updates"][str(len(fit_rows) // 11)])
    elif max_updates is not None:
        updates = max_updates
    else:
        raise ValueError("few-shot fit count must match registered K except in smoke mode")
    if max_updates is not None:
        updates = min(updates, max_updates)
    warmup = max(1, int(float(config["warmup_ratio"]) * updates))
    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    order = list(range(len(fit_rows)))
    random.Random(seed + 801).shuffle(order)
    trace = []
    model.train()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        sft_values, pool_values = [], []
        for micro in range(accumulation):
            index = step * accumulation + micro
            if mode == "full" and index >= len(order):
                break
            row = fit_rows[order[index % len(order)]]
            if mode == "full":
                target = str(row["leaf_id"])
            else:
                observations = teacher[row["sample_id"]]
                if tuple(observations) != OBSERVATION_KEYS:
                    raise ValueError("teacher observation schema/key order changed")
                target = json.dumps({"observations": observations, "class_id": row["leaf_id"]},
                                    ensure_ascii=False, separators=(",", ":"))
            batch, prefix_length = _batch(processor, row, prompt, target, config)
            sft, pool = _losses(model, batch, prefix_length, head, row["leaf_id"], objective)
            ((sft + pool) / accumulation).backward()
            sft_values.append(float(sft.detach().cpu()))
            pool_values.append(float(pool.detach().cpu()))
        active_grads = [p.grad for p in trainable if p.grad is not None]
        if not active_grads or any(not torch.isfinite(g).all() for g in active_grads):
            raise RuntimeError("no finite trainable gradients at adaptation update")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        trace.append({"update": step + 1, "response_loss": float(np.mean(sft_values)),
                      "poolce_loss": float(np.mean(pool_values))})
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter", safe_serialization=True)
    np.savez_compressed(output_dir / "direct_head.npz", **{key: value.detach().cpu().numpy() for key, value in head.items()})
    fit_features = extract_features(model, processor, fit_rows, prompt, config)
    query_features = extract_features(model, processor, query_rows, prompt, config)
    readout = fit_logistic(fit_features["lm_mean"], labels, c=1.0 if mode == "full" else 0.01)
    task_prob = predict_logistic(readout, query_features["lm_mean"])
    direct_logits = ((query_features["lm_mean"] - head["mean"].detach().cpu().numpy()) /
                     head["scale"].detach().cpu().numpy()) @ head["weight"].detach().cpu().numpy().T + head["bias"].detach().cpu().numpy()
    direct_logits -= direct_logits.max(axis=1, keepdims=True)
    direct_prob = np.exp(direct_logits); direct_prob /= direct_prob.sum(axis=1, keepdims=True)
    np.savez_compressed(output_dir / "features.npz", sample_id=np.array([r["sample_id"] for r in fit_rows]),
                        labels=labels, **{f"fit_{k}": v for k, v in fit_features.items()},
                        **{f"query_{k}": v for k, v in query_features.items()})
    np.savez_compressed(output_dir / "predictions.npz", sample_id=np.array([r["sample_id"] for r in query_rows]),
                        task_probabilities=task_prob, direct_probabilities=direct_prob)
    report = {"mode": mode, "objective": objective, "seed": seed, "fit_rows": len(fit_rows),
              "query_rows": len(query_rows), "updates": updates, "accumulation": accumulation,
              "model_ref": model_ref, "trace": trace,
              "paper_reproduction": max_updates is None}
    (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
