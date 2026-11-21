"""Command line entrypoints for the paper's fixed data and method protocols."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .baselines import train_convnext, train_hog_svm, train_yolo
from .audit import retained_query_ids, source_components, write_retained
from .data import assert_group_disjoint, load_manifest, rows_by_id, select_rows, validate_historical
from .fusion import EntropyGate, fixed_fewshot, full_ensemble
from .metrics import paired_score
from .readout import fit_frozen_partner, fit_logistic, frozen_qwen3_ridge, predict_logistic
from .reference import frozen_partner
from .teacher import generate, load_teacher

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _label_free(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in {"leaf_id", "leaf_name", "split", "sha256"}}
            for row in rows]


def _smoke_subset(rows: list[dict], n: int | None) -> list[dict]:
    if n is None:
        return rows
    selected = []
    counts = Counter()
    for row in rows:
        if counts[row["leaf_id"]] < n:
            selected.append(row); counts[row["leaf_id"]] += 1
    if len(selected) != 11 * n:
        raise ValueError("smoke subset lacks requested examples in each class")
    return selected


def _cohort(rows: list[dict], name: str, provenance: Path) -> list[dict]:
    if name == "full_test":
        return [r for r in rows if r["split"] == "test"]
    query = [r for r in _csv(provenance / "fewshot_queries.csv") if r["cohort"] == name]
    if not query:
        raise ValueError(f"unknown cohort {name}")
    return select_rows(rows, [r["sample_id"] for r in query])


def _fit_rows(rows: list[dict], provenance: Path, mode: str, member: str | None, draw: int, k: int) -> list[dict]:
    if mode == "full":
        ids = [r["sample_id"] for r in _csv(provenance / "full_members.csv") if r["member_id"] == member]
    else:
        ids = [r["sample_id"] for r in _csv(provenance / "fewshot_supports.csv")
               if int(r["draw"]) == draw and int(r["k"]) == k]
    if not ids:
        raise ValueError("requested fixed fit membership not found")
    return select_rows(rows, ids)


def _inputs(args):
    rows = load_manifest(args.manifest, args.data_root, verify_images=args.verify_images)
    validate_historical(rows)
    provenance = args.provenance
    fit = _fit_rows(rows, provenance, args.mode, args.member, args.draw, args.k)
    query = _cohort(rows, args.cohort, provenance)
    fit = _smoke_subset(fit, args.smoke_per_class)
    query = _smoke_subset(query, args.smoke_queries_per_class)
    assert_group_disjoint(fit, query)
    return fit, query


def _reference_predictions(fit: list[dict], query: list[dict], config: dict, args, output: Path):
    features = frozen_partner(fit + query, args.qwen2_model or config["qwen2_model"],
                              args.dino_model or config["dinov2_model"], config["fewshot_prompt"])
    labels = np.array([r["leaf_id"] for r in fit])
    groups = np.array([r["group_id"] for r in fit])
    readout, report = fit_frozen_partner(features[:len(fit)], labels, groups, seed=args.seed)
    prediction = predict_logistic(readout, features[len(fit):])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reference.npz", sample_id=np.array([r["sample_id"] for r in query]),
                        probabilities=prediction)
    _write_json(output / "reference.json", report)
    return prediction, features


def _recorded_support_folds(fit: list[dict], provenance: Path, draw: int, k: int):
    records = [r for r in _csv(provenance / "fewshot_oof_folds.csv")
               if int(r["draw"]) == draw and int(r["k"]) == k]
    position = {r["sample_id"]: i for i, r in enumerate(fit)}
    folds = []
    validation_seen = []
    for fold in range(2):
        subset = [r for r in records if int(r["fold"]) == fold]
        train_ids = [r["sample_id"] for r in subset if r["role"] == "train"]
        valid_ids = [r["sample_id"] for r in subset if r["role"] == "validation"]
        if set(train_ids) | set(valid_ids) != set(position) or set(train_ids) & set(valid_ids):
            raise ValueError("recorded support OOF fold does not partition current supports")
        tr, va = np.array([position[sid] for sid in train_ids]), np.array([position[sid] for sid in valid_ids])
        assert_group_disjoint([fit[i] for i in tr], [fit[i] for i in va])
        if {fit[i]["leaf_id"] for i in tr} != set(range(11)):
            raise ValueError("recorded support OOF fit lacks a class")
        folds.append((tr, va))
        validation_seen.extend(valid_ids)
    if len(validation_seen) != len(position) or set(validation_seen) != set(position):
        raise ValueError("recorded support OOF validation coverage is incomplete")
    return folds


def _run_full(args) -> None:
    from .qwen import train

    config = _config(args.config)
    fit, query = _inputs(args)
    report = train(args.qwen3_model or config["qwen3_model"], config, fit, _label_free(query), args.output,
                   mode="full", objective=args.objective, seed=args.seed,
                   max_updates=args.max_updates)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _fuse_full(args) -> None:
    task_members, reference_members, ids = [], [], None
    for index, member_dir in enumerate(args.members):
        with np.load(member_dir / "features.npz", allow_pickle=False) as z:
            labels = z["labels"].astype(int)
            if index == 2:
                fit_visual = np.concatenate([z["fit_visual_mean"], z["fit_visual_max"]], axis=1)
                query_visual = np.concatenate([z["query_visual_mean"], z["query_visual_max"]], axis=1)
                for matrix in (fit_visual, query_visual):
                    half = matrix.shape[1] // 2
                    matrix[:, :half] /= np.maximum(np.linalg.norm(matrix[:, :half], axis=1, keepdims=True), 1e-12)
                    matrix[:, half:] /= np.maximum(np.linalg.norm(matrix[:, half:], axis=1, keepdims=True), 1e-12)
            else:
                fit_visual, query_visual = z["fit_visual_mean"], z["query_visual_mean"]
            readout = fit_logistic(fit_visual, labels, c=0.1)
            reference_members.append(predict_logistic(readout, query_visual))
        with np.load(member_dir / "predictions.npz", allow_pickle=False) as z:
            current_ids = z["sample_id"].astype(str).tolist()
            if ids is not None and current_ids != ids:
                raise ValueError("full-data member query order differs")
            ids = current_ids
            task_members.append(z["task_probabilities"])
    probabilities = full_ensemble(np.stack(task_members), np.stack(reference_members))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_id=np.array(ids), probabilities=probabilities,
                        task_members=np.stack(task_members), reference_members=np.stack(reference_members))
    print(f"saved {len(ids)} full-data fused predictions to {args.output}")


def _run_fewshot(args) -> None:
    from .qwen import train

    config = _config(args.config)
    fit, query = _inputs(args)
    teacher = load_teacher(args.teacher_cache)
    if not set(r["sample_id"] for r in fit).issubset(teacher):
        raise ValueError("teacher cache does not cover exact current support")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume_adapted:
        train(args.qwen3_model or config["qwen3_model"], config, fit, _label_free(query), args.output / "adapted",
              mode="fewshot", objective=args.objective, teacher=teacher, seed=args.seed,
              max_updates=args.max_updates)
    else:
        with np.load(args.output / "adapted/predictions.npz", allow_pickle=False) as z:
            if z["sample_id"].astype(str).tolist() != [r["sample_id"] for r in query]:
                raise ValueError("saved adapted predictions do not match current query IDs")
        with np.load(args.output / "adapted/features.npz", allow_pickle=False) as z:
            if z["sample_id"].astype(str).tolist() != [r["sample_id"] for r in fit]:
                raise ValueError("saved adapted features do not match current support IDs")
    reference_prob, all_features = _reference_predictions(fit, query, config, args, args.output)
    with np.load(args.output / "adapted/predictions.npz", allow_pickle=False) as z:
        task_prob = z["direct_probabilities"]
    if args.k == 5 or args.smoke_per_class is not None:
        fusion = fixed_fewshot(task_prob, reference_prob)
        gate_report = {"mode": "fixed_raw", "task_weight": 0.25}
    else:
        y = np.array([r["leaf_id"] for r in fit])
        groups = np.array([r["group_id"] for r in fit])
        folds = _recorded_support_folds(fit, args.provenance, args.draw, args.k)
        task_oof = np.full((len(fit), 11), np.nan)
        reference_oof = np.full((len(fit), 11), np.nan)
        for fold, (tr, va) in enumerate(folds):
            fold_fit = [fit[i] for i in tr]
            fold_query = [fit[i] for i in va]
            assert_group_disjoint(fold_fit, fold_query)
            fold_dir = args.output / f"support_oof/fold_{fold}"
            train(args.qwen3_model or config["qwen3_model"], config, fold_fit, _label_free(fold_query), fold_dir,
                  mode="fewshot", objective=args.objective, teacher=teacher,
                  seed=args.seed + fold + 1, max_updates=args.max_updates)
            with np.load(fold_dir / "predictions.npz", allow_pickle=False) as z:
                task_oof[va] = z["direct_probabilities"]
            reference_readout, _ = fit_frozen_partner(all_features[tr], y[tr], groups[tr], seed=args.seed + fold + 1)
            reference_oof[va] = predict_logistic(reference_readout, all_features[va])
        if not np.isfinite(task_oof).all() or not np.isfinite(reference_oof).all():
            raise RuntimeError("support OOF coverage incomplete")
        gate = EntropyGate.fit(task_oof, reference_oof, y)
        fusion = gate.predict(task_prob, reference_prob)
        gate_report = {"mode": "entropy_gate", "task_temperature": gate.task_temperature,
                       "reference_temperature": gate.reference_temperature, "beta": gate.beta}
        np.savez_compressed(args.output / "support_oof.npz",
                            sample_id=np.array([r["sample_id"] for r in fit]),
                            task_probabilities=task_oof, reference_probabilities=reference_oof)
    np.savez_compressed(args.output / "fused.npz", sample_id=np.array([r["sample_id"] for r in query]),
                        task_probabilities=task_prob, reference_probabilities=reference_prob,
                        fixed_raw=fixed_fewshot(task_prob, reference_prob), probabilities=fusion)
    _write_json(args.output / "fusion.json", gate_report)
    print(json.dumps({"fit": len(fit), "query": len(query), **gate_report}, indent=2))


def _run_baseline(args) -> None:
    rows = load_manifest(args.manifest, args.data_root, verify_images=args.verify_images)
    validate_historical(rows)
    fit = _smoke_subset([r for r in rows if r["split"] == "train"], args.smoke_per_class)
    query = _smoke_subset(_cohort(rows, args.cohort, args.provenance), args.smoke_queries_per_class)
    assert_group_disjoint(fit, query)
    epochs = args.epochs if args.epochs is not None else (8 if args.method == "yolo" else 6)
    if args.method == "hog":
        report = train_hog_svm(fit, _label_free(query), args.output, args.seed)
    elif args.method == "convnext":
        report = train_convnext(fit, _label_free(query), args.output,
                                seed=args.seed, epochs=epochs, pretrained=not args.no_pretrained)
    else:
        if not args.yolo_checkpoint:
            raise ValueError("--yolo-checkpoint is required for YOLO")
        report = train_yolo(fit, _label_free(query), args.output,
                            checkpoint=args.yolo_checkpoint, seed=args.seed, epochs=epochs)
    _write_json(args.output / "run.json", report)
    print(json.dumps(report, indent=2))


def _teacher(args) -> None:
    fit, _ = _inputs(args)
    config = _config(args.config)
    generate(_label_free(fit), args.qwen3_model or config["qwen3_model"], args.output)


def _frozen_ridge(args) -> None:
    from .qwen import extract_features, load_base
    import torch

    config = _config(args.config)
    fit, query = _inputs(args)
    model, processor = load_base(args.qwen3_model or config["qwen3_model"], config)
    features = extract_features(model, processor, fit + _label_free(query), config["fewshot_prompt"], config, visual=False)["lm_mean"]
    probabilities = frozen_qwen3_ridge(features[:len(fit)], np.array([r["leaf_id"] for r in fit]), features[len(fit):])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_id=np.array([r["sample_id"] for r in query]), probabilities=probabilities)
    del model
    torch.cuda.empty_cache()


def _audit(args) -> None:
    from .data import read_ids

    rows = load_manifest(args.manifest)
    components = source_components(rows, args.accepted_links)
    query_ids = set(read_ids(args.queries))
    fit_ids = set(read_ids(args.fit_or_selection))
    unresolved = set(read_ids(args.unresolved))
    retained, report = retained_query_ids(rows, components, query_ids, fit_ids, unresolved)
    write_retained(args.output, retained, components)
    report["accepted_links_file"] = args.accepted_links.name
    report["unresolved_file"] = args.unresolved.name
    _write_json(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


def _score(args) -> None:
    rows = load_manifest(args.manifest)
    label_by_id = {r["sample_id"]: r["leaf_id"] for r in rows}
    with np.load(args.predictions, allow_pickle=False) as z:
        ids = z["sample_id"].astype(str).tolist()
        probabilities = z[args.probability_key]
    keep = None
    if args.retained_ids:
        keep = {r["sample_id"] for r in _csv(args.retained_ids)}
        if len(keep) != len(_csv(args.retained_ids)):
            raise ValueError("retained cohort has duplicate IDs")
    report = paired_score(label_by_id, ids, probabilities, keep)
    if args.expected_rows is not None and report["n"] != args.expected_rows:
        raise ValueError(f"cohort count {report['n']} differs from expected {args.expected_rows}")
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="pacr")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("validate-data")
    data.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/manifests/insplad_clean_v1.csv")
    data.add_argument("--data-root", type=Path)
    data.add_argument("--verify-images", action="store_true")

    def common(command, mode):
        p = sub.add_parser(command)
        p.set_defaults(mode=mode)
        p.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/manifests/insplad_clean_v1.csv")
        p.add_argument("--provenance", type=Path, default=PACKAGE_ROOT / "data/provenance")
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--verify-images", action="store_true")
        p.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/paper.json")
        p.add_argument("--qwen3-model", required=False)
        p.add_argument("--qwen2-model", required=False)
        p.add_argument("--dino-model", required=False)
        p.add_argument("--member", choices=("member0", "member1", "member2"), default="member0")
        p.add_argument("--draw", type=int, choices=(0, 1, 2), default=0)
        p.add_argument("--k", type=int, choices=(5, 10, 20), default=5)
        p.add_argument("--cohort", choices=("full_test", "development", "fewshot_test"),
                       default="full_test" if mode == "full" else "development")
        p.add_argument("--seed", type=int, default=20261301)
        p.add_argument("--smoke-per-class", type=int)
        p.add_argument("--smoke-queries-per-class", type=int)
        p.add_argument("--max-updates", type=int)
        return p

    full = common("train-full", "full")
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--objective", choices=("response", "poolce", "joint"), default="joint")
    fuse = sub.add_parser("fuse-full")
    fuse.add_argument("--members", type=Path, nargs=3, required=True)
    fuse.add_argument("--output", type=Path, required=True)
    few = common("train-fewshot", "fewshot")
    few.add_argument("--output", type=Path, required=True)
    few.add_argument("--teacher-cache", type=Path, required=True)
    few.add_argument("--objective", choices=("response", "poolce", "joint"), default="joint")
    few.add_argument("--resume-adapted", action="store_true")
    teacher = common("generate-teacher", "fewshot")
    teacher.add_argument("--output", type=Path, required=True)
    ridge = common("frozen-qwen3-ridge", "fewshot")
    ridge.add_argument("--output", type=Path, required=True)
    baseline = common("baseline", "full")
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--method", choices=("hog", "convnext", "yolo"), required=True)
    baseline.add_argument("--epochs", type=int)
    baseline.add_argument("--no-pretrained", action="store_true")
    baseline.add_argument("--yolo-checkpoint")
    scoring = sub.add_parser("score")
    scoring.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/manifests/insplad_clean_v1.csv")
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--probability-key", default="probabilities")
    scoring.add_argument("--retained-ids", type=Path)
    scoring.add_argument("--expected-rows", type=int)
    scoring.add_argument("--output", type=Path)
    audit = sub.add_parser("audit-cohort")
    audit.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/manifests/insplad_clean_v1.csv")
    audit.add_argument("--accepted-links", type=Path, required=True)
    audit.add_argument("--queries", type=Path, required=True)
    audit.add_argument("--fit-or-selection", type=Path, required=True)
    audit.add_argument("--unresolved", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-data":
        rows = load_manifest(args.manifest, args.data_root, verify_images=args.verify_images)
        print(json.dumps(validate_historical(rows), indent=2))
    elif args.command == "train-full":
        _run_full(args)
    elif args.command == "fuse-full":
        _fuse_full(args)
    elif args.command == "train-fewshot":
        _run_fewshot(args)
    elif args.command == "generate-teacher":
        _teacher(args)
    elif args.command == "frozen-qwen3-ridge":
        _frozen_ridge(args)
    elif args.command == "baseline":
        _run_baseline(args)
    elif args.command == "score":
        _score(args)
    elif args.command == "audit-cohort":
        _audit(args)


if __name__ == "__main__":
    main()
