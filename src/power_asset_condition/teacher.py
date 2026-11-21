"""Support-only eight-field visual observation generation."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from . import OBSERVATION_KEYS

SYSTEM = "You are a visual inspection evidence recorder. Describe only directly visible evidence. Do not classify the image and do not infer a final label."
USER = "Inspect the image and return exactly one JSON object with these eight array fields in this order: equipment, visible_components, structural_state, surface_state, missing_or_broken_parts, abnormal_visual_cues, spatial_relations, uncertain_observations. Use short factual phrases and an empty list when evidence is absent. Do not output class labels, dataset metadata, file paths, confidence scores, or explanations outside JSON."
TEACHER_IMAGE_MAX_PIXELS = 1024 * 1024


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
    existing_records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()] if output.exists() else []
    if any(record.get("image_max_pixels") != TEACHER_IMAGE_MAX_PIXELS for record in existing_records):
        raise ValueError("existing teacher cache has an unknown image limit; choose a new output path")
    existing = load_teacher(output) if output.exists() else {}
    requested = {row["sample_id"] for row in rows}
    if not set(existing).issubset(requested):
        raise ValueError("existing teacher cache contains IDs outside the requested supports")
    pending = [row for row in rows if row["sample_id"] not in existing]
    if not pending:
        return
    processor = AutoProcessor.from_pretrained(model_ref)
    size = getattr(processor.image_processor, "size", None)
    if not isinstance(size, dict) or "longest_edge" not in size:
        raise RuntimeError("Qwen3-VL processor has no longest_edge size limit")
    size["longest_edge"] = TEACHER_IMAGE_MAX_PIXELS
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_ref, dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0}, attn_implementation="sdpa").eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for row in pending:
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
            for token_cap in (384, 768, 1024):
                generated = model.generate(**batch, max_new_tokens=token_cap, do_sample=False)
                answer = processor.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                try:
                    observations = parse_observations(answer)
                    break
                except (json.JSONDecodeError, ValueError) as error:
                    if token_cap == 1024:
                        raise RuntimeError(f"teacher output for {row['sample_id']} is invalid") from error
            record = {"sample_id": row["sample_id"], "observations": observations,
                      "image_max_pixels": TEACHER_IMAGE_MAX_PIXELS}
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()


def load_teacher(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    teacher = {record["sample_id"]: record["observations"] for record in records}
    if len(teacher) != len(records) or any(tuple(obs) != OBSERVATION_KEYS for obs in teacher.values()):
        raise ValueError("teacher cache has duplicate IDs or wrong schema")
    return teacher
