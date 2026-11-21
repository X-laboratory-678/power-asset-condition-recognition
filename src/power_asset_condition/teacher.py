"""Support-only eight-field visual observation generation."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from . import OBSERVATION_KEYS

SYSTEM = "You are a visual inspection evidence recorder. Describe only directly visible evidence. Do not classify the image and do not infer a final label."
USER = "Inspect the image and return exactly one JSON object with these eight array fields in this order: equipment, visible_components, structural_state, surface_state, missing_or_broken_parts, abnormal_visual_cues, spatial_relations, uncertain_observations. Use short factual phrases and an empty list when evidence is absent. Do not output class labels, dataset metadata, file paths, confidence scores, or explanations outside JSON."


def parse_observations(raw: str) -> dict[str, list[str]]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:].removesuffix("```").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    observations = {}
    for key in OBSERVATION_KEYS:
        items = value.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"teacher field {key} must be a string array")
        observations[key] = items[:8]
    return observations


def generate(rows: list[dict], model_ref: str, output: Path) -> None:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if any("leaf_id" in row or "label_id" in row for row in rows):
        raise ValueError("teacher generation accepts only label-free support rows")
    processor = AutoProcessor.from_pretrained(model_ref)
    if not hasattr(processor.image_processor, "max_pixels"):
        raise RuntimeError("Qwen3-VL processor has no max_pixels field")
    processor.image_processor.max_pixels = 1024 * 1024
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_ref, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0}, attn_implementation="sdpa").eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": USER}]},
            ]
            batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt")
            batch = {k: (v.to("cuda:0", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda:0"))
                     for k, v in batch.items() if hasattr(v, "to")}
            generated = model.generate(**batch, max_new_tokens=384, do_sample=False)
            answer = processor.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            record = {"sample_id": row["sample_id"], "observations": parse_observations(answer)}
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_teacher(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    teacher = {record["sample_id"]: record["observations"] for record in records}
    if len(teacher) != len(records) or any(tuple(obs) != OBSERVATION_KEYS for obs in teacher.values()):
        raise ValueError("teacher cache has duplicate IDs or wrong schema")
    return teacher
