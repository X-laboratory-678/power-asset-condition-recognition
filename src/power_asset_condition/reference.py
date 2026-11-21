"""Frozen Qwen2-VL and DINOv2 image features for few-shot comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .readout import qwen2_dino_features


def extract_qwen2(rows: list[dict], model_ref: str, prompt: str, max_pixels: int = 262144) -> np.ndarray:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_ref)
    processor.image_processor.size.longest_edge = max_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_ref, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    result = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            user = {"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            batch = processor.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt")
            batch = {k: (v.to("cuda:0", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda:0"))
                     for k, v in batch.items() if hasattr(v, "to")}
            mask = batch["input_ids"][0].eq(int(model.config.image_token_id))
            hidden = model.model(**batch, use_cache=False).last_hidden_state
            vector = hidden[0, mask].float().mean(0).cpu().numpy()
            if vector.shape != (3584,) or not np.isfinite(vector).all():
                raise RuntimeError("invalid Qwen2-VL 3584-wide image mean")
            result.append(vector)
    del model
    torch.cuda.empty_cache()
    return np.stack(result).astype(np.float32)


def extract_dino(rows: list[dict], model_ref: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_ref)
    model = AutoModel.from_pretrained(model_ref).to("cuda:0").eval()
    cls, patch = [], []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            batch = {k: v.to("cuda:0") for k, v in processor(images=image, return_tensors="pt").items()}
            hidden = model(**batch).last_hidden_state.float()
            cls.append(hidden[0, 0].cpu().numpy())
            patch.append(hidden[0, 1:].mean(0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(cls).astype(np.float32), np.stack(patch).astype(np.float32)


def frozen_partner(rows: list[dict], qwen2_model: str, dino_model: str, prompt: str) -> np.ndarray:
    qwen2 = extract_qwen2(rows, qwen2_model, prompt)
    cls, patch = extract_dino(rows, dino_model)
    return qwen2_dino_features(qwen2, cls, patch)
