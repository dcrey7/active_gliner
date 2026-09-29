"""One frozen-pool acquisition and LoRA training run."""

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field, model_validator

from active_gliner import data, mixing, model, pins, protocol, selection, tasks
from active_gliner.evaluate.calibration import calibration_rows
from active_gliner.evaluate.errors import error_rows
from active_gliner.evaluate.reports import per_type_rows, write_csv
from active_gliner.observe.logs import write_json, write_jsonl
from active_gliner.observe.plots import plot_calibration, plot_training
from active_gliner.report import write_report
from active_gliner.teachers import TeacherConfig, label_records, prompts, validate
from active_gliner.teachers.config import REPO_ROOT
from active_gliner.train import train


class RunConfig(BaseModel):
    dataset: str
    block: str = "adhoc"
    gt_fraction: float | None = Field(default=None, ge=0, le=1)
    gt_assignment: Literal["routed", "random"] | None = None
    no_prediction: Literal["zero", "last"] = "zero"
    recipe_file: str | None = None
    recipe_hash: str | None = None
    locale: str = "en-US"
    # avg, mse and mnlp are the thesis selectors (selection.THESIS_KEYS).
    selector: Literal["min", "random", "diversity", "all", "avg", "mse", "mnlp"] = "min"
    n: int = Field(gt=0)
    seed: int = Field(ge=0)
    labels_source: str = "ground_truth"
    max_steps: int = Field(default=1000, gt=0)
    eval_steps: int = Field(default=100, gt=0)
    batch_size: int = Field(default=8, gt=0)
    grad_accum: int = Field(default=1, gt=0)
    task_lr: float = Field(default=5e-4, gt=0)
    lora_r: int = Field(default=8, gt=0)
    lora_alpha: float = Field(default=16.0, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    lora_targets: list[str] = Field(default_factory=lambda: ["encoder", "all_task_heads"])
    # "full" trains every weight (thesis E2); encoder_lr applies only then.
    finetune: Literal["lora", "full"] = "lora"
    encoder_lr: float = Field(default=1e-5, gt=0)
    # Names a training variant in the folder (for example "heads-only"), so it cannot
    # overwrite the main run with the same selector, N and seed.
    variant: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    augmentation: bool = True
    # Research runs use the dev-selected value from the frozen recipe.
    relation_head: Literal["classifier", "native", "marker"] = "classifier"
    relation_neg_ratio: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    acq_threshold: float = Field(default=0.5, ge=0, le=1)
    eval_threshold: float = Field(default=0.5, ge=0, le=1)
    slot_mode: Literal["structure", "entity"] = "structure"
    early_stopping_patience: int = Field(default=3, gt=0)
    # Deterministic kernels, fp32 and no TF32: slower, but a seed repeats exactly.
    exact_numerics: bool = False
    pool_limit: int | None = Field(default=None, gt=0)
    dev_limit: int | None = Field(default=None, gt=0)
    test_limit: int | None = Field(default=None, gt=0)
    out_root: str = "runs"

    @model_validator(mode="after")
    def validate_mixing(self):
        if self.dataset == "crossre":
            self.augmentation = False
            if self.relation_head != "classifier":
                self.lora_targets = ["encoder"] + (
                    ["relation_scorer"] if self.relation_head == "native" else []
                )
        if self.gt_fraction is not None:
            if self.labels_source == "ground_truth" or self.gt_assignment is None:
                raise ValueError("Mixing requires a teacher and a gold assignment")
        elif self.gt_assignment is not None:
            raise ValueError("Gold assignment requires gt_fraction")
        return self


def folder_name(cfg: RunConfig) -> str:
    name = (
        f"all-seed{cfg.seed}"
        if cfg.selector == "all"
        else f"{cfg.selector}-N{cfg.n}-seed{cfg.seed}"
    )
    if cfg.gt_fraction is not None:
        name += f"-gt{cfg.gt_fraction * 100:g}{cfg.gt_assignment}"
    if cfg.no_prediction == "last":
        name += "-nopredlast"
    if cfg.variant:
        name += f"-{cfg.variant}"
    return name


def experiment_dir(cfg: RunConfig, out_root=None) -> Path:
    task_name = tasks.for_dataset(cfg.dataset).__name__.rsplit(".", 1)[-1]
    return (
        Path(out_root if out_root is not None else cfg.out_root)
        / task_name
        / cfg.dataset
        / cfg.locale
        / cfg.labels_source
        / folder_name(cfg)
    )


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def run_experiment(cfg: RunConfig) -> Path:
    import yaml

    if cfg.acq_threshold != 0.5:
        raise ValueError("The research acquisition threshold is frozen at 0.5")
    started = time.monotonic()
    teacher = None
    if cfg.labels_source != "ground_truth":
        teacher = TeacherConfig.from_yaml(
            REPO_ROOT / "configs/teachers" / f"{cfg.labels_source}.yaml"
        )
    prompt_version = prompts.version_for(cfg.dataset)
    definitions = prompts.prompt_for(cfg.dataset, prompt_version)
    task = tasks.for_dataset(cfg.dataset)
    splits = data.load(cfg.dataset, cfg.locale)
    labels = tasks.labels(cfg.dataset, splits)
    task_name = splits["pool"][0].task
    seeds = {
        name: cfg.seed * 1000 + offset
        for offset, name in enumerate(("selection", "data_order", "lora_init"), start=1)
    }
    run_dir = experiment_dir(cfg)
    # Refuse to mix new logs with a previous run or destroy a saved adapter.
    run_dir.mkdir(parents=True, exist_ok=False)
    data.write_split_ids(splits, cfg.dataset, cfg.locale, Path(cfg.out_root) / "splits")
    splits = {name: records[: getattr(cfg, f"{name}_limit")] for name, records in splits.items()}
    if any(not records for records in splits.values()):
        raise ValueError("Pool, dev and test must each contain records")
    if cfg.selector != "all" and cfg.n > len(splits["pool"]):
        raise ValueError("n exceeds the pool size")
    resolved = cfg.model_dump()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({**resolved, "seeds": seeds}), encoding="utf-8"
    )
    write_json(
        run_dir / "hashes.json",
        {
            "split": data.split_hash(splits),
            "schema": _hash(
                {"labels": sorted(labels.values()), "task": task_name, "slot_mode": cfg.slot_mode}
            ),
            "config": _hash(resolved),
            **({"prompt": prompts.prompt_hash(task_name, labels, definitions)} if teacher else {}),
        },
    )
    revisions = {cfg.dataset: data.dataset_pin(cfg.dataset)}
    pins.write_pins(
        run_dir / "pins.json",
        pins.collect_pins(
            model.STUDENT_REPO,
            model.STUDENT_SHA,
            revisions,
            extra={
                "split_sizes": {name: len(records) for name, records in splits.items()},
                **(
                    {
                        "embedding_model": {
                            "repo": model.STUDENT_REPO,
                            "sha": model.STUDENT_SHA,
                            "component": "student encoder",
                            "pooling": "mean sentence tokens",
                        }
                    }
                    if cfg.selector == "diversity"
                    else {}
                ),
            },
        ),
    )
    student = model.load_student(device="cuda")
    if task_name == "relations" and cfg.relation_head != "classifier":
        from active_gliner.tasks.relation_heads import configure

        configure(student, cfg.relation_head, labels)
    student.eval()
    selected_ids, confidences = score_pool(
        student, cfg, seeds, run_dir, task, labels, splits["pool"]
    )
    acquisition = json.loads((run_dir / "acquisition.json").read_text())
    write_json(
        run_dir / "protocol.json",
        {
            "protocol_fingerprint": protocol.protocol_fingerprint(
                cfg,
                acquisition["pool_score_hash"],
                prompts.prompt_hash(task_name, labels, definitions),
            )
        },
    )
    ids = [record.id for record in splits["pool"]]
    gold = set()
    if cfg.gt_fraction is not None:
        gold = mixing.gold_ids(
            selected_ids,
            dict(zip(ids, confidences, strict=True)),
            cfg.gt_fraction,
            cfg.gt_assignment,
            seeds["selection"],
        )
        write_json(run_dir / "gold_ids.json", sorted(gold))
    pool = {record.id: record for record in splits["pool"]}
    selected = [pool[item_id] for item_id in selected_ids]
    if teacher:
        labelled = label_records(
            [record for record in selected if record.id not in gold],
            task_name,
            labels,
            teacher,
            REPO_ROOT / "labels",
            dataset=cfg.dataset,
            definitions=definitions,
            prompt_version=prompt_version,
        )
        write_json(run_dir / "teacher_stats.json", labelled.stats)
        teacher_records = {
            r.id: r
            for r in validate.teacher_records(
                [record for record in selected if record.id not in gold], labelled.labels
            )
        }
        selected = [r if r.id in gold else teacher_records[r.id] for r in selected]
        run_pins = json.loads((run_dir / "pins.json").read_text())
        run_pins["teacher"] = {
            **teacher.model_dump(),
            "prompt_version": prompt_version,
            "prompt_hash": prompts.prompt_hash(task_name, labels, definitions),
            "label_cache_fingerprint": _hash(labelled.labels),
        }
        write_json(run_dir / "pins.json", run_pins)
    training = train(
        student,
        cfg,
        seeds,
        run_dir,
        task,
        labels,
        selected,
        splits["dev"],
    )
    del student
    gc.collect()
    torch.cuda.empty_cache()
    return finish_run(
        cfg,
        seeds,
        run_dir,
        task,
        labels,
        training,
        {name: splits[name] for name in ("dev", "test")},
        len(selected_ids),
        started,
    )


def score_pool(
    student, cfg, seeds, run_dir, task, labels, pool_records
) -> tuple[list[str], list[float]]:
    artifact = protocol.pool_score_path(cfg, labels, pool_records)
    items_artifact = protocol.pool_items_path(artifact)
    with protocol.artifact_lock(artifact):
        if not artifact.exists() or not items_artifact.exists():
            with protocol.inference_precision():
                predictions = task.predict(
                    student,
                    pool_records,
                    labels,
                    threshold=cfg.acq_threshold,
                    batch_size=32,
                    slot_mode=cfg.slot_mode,
                )
        # Never rewrite an existing score file: its hash identifies the run protocol.
        if not artifact.exists():
            scores = [
                {
                    "id": p["id"],
                    "confidence": p["confidence"],
                    "has_prediction": bool(task.pred_items(p)),
                    **({"pair_count": len(p["pairs"])} if "pairs" in p else {}),
                }
                for p in predictions
            ]
            temporary = artifact.with_suffix(".tmp")
            write_jsonl(temporary, scores)
            temporary.replace(artifact)
        if not items_artifact.exists():
            items = [{"id": p["id"], "items": selection.item_confidences(p)} for p in predictions]
            temporary = items_artifact.with_suffix(".tmp")
            write_jsonl(temporary, items)
            temporary.replace(items_artifact)
        scores = [json.loads(line) for line in artifact.read_text().splitlines()]
    if [p["id"] for p in scores] != [r.id for r in pool_records]:
        raise ValueError("Pool score artifact IDs do not match the frozen pool")
    score_hash = protocol.file_hash(artifact)
    ids = [p["id"] for p in scores]
    confidences = [p["confidence"] for p in scores]
    if cfg.no_prediction == "last":
        confidences = selection.rank_empty_last(confidences, [p["has_prediction"] for p in scores])
    if cfg.no_prediction == "last":
        for row, confidence in zip(scores, confidences, strict=True):
            row["selection_confidence"] = confidence
    write_jsonl(run_dir / "pool_scores.jsonl", scores)
    if cfg.selector == "all":
        selected_ids = ids
    elif cfg.selector == "diversity":
        embeddings = pool_embeddings(student, cfg, pool_records)
        selected_ids = selection.select_diverse(ids, embeddings, cfg.n, seeds["selection"])
    elif cfg.selector in selection.THESIS_KEYS:
        rows = [json.loads(line) for line in items_artifact.read_text().splitlines()]
        if [row["id"] for row in rows] != ids:
            raise ValueError("Pool item artifact IDs do not match the frozen pool")
        keys = selection.thesis_keys(cfg.selector, [row["items"] for row in rows])
        selected_ids = selection.select(ids, keys, cfg.n, cfg.selector, seeds["selection"])
    else:
        selected_ids = selection.select(ids, confidences, cfg.n, cfg.selector, seeds["selection"])
    selection_hash = protocol.assert_selection(cfg, score_hash, selected_ids)
    write_json(
        run_dir / "acquisition.json",
        {
            "pool_score_hash": score_hash,
            "pool_score_artifact": str(artifact),
            "selection_fingerprint": selection_hash,
        },
    )
    write_json(run_dir / "selected_ids.json", selected_ids)
    return selected_ids, confidences


def pool_embeddings(student, cfg, pool_records):
    """Compute pool embeddings once and share the file with every paired arm."""
    import numpy as np

    artifact = protocol.embedding_path(cfg, pool_records)
    with protocol.artifact_lock(artifact):
        if not artifact.exists():
            with protocol.inference_precision():
                values = model.sentence_embeddings(student, [r.text for r in pool_records], 32)
            temporary = artifact.with_name(artifact.name + ".tmp")
            with open(temporary, "wb") as handle:
                np.save(handle, np.asarray(values, dtype=np.float32))
            temporary.replace(artifact)
        values = np.load(artifact)
    if len(values) != len(pool_records):
        raise ValueError("Pool embedding artifact does not match the frozen pool")
    return values


def choose_eval_threshold(scores: dict[float, float]) -> float:
    """Choose by dev micro-F1, then distance to 0.5, then lower threshold."""
    if set(scores) != {0.3, 0.4, 0.5, 0.6, 0.7}:
        raise ValueError("Expected the frozen five-threshold dev grid")
    return min(scores, key=lambda t: (-scores[t], round(abs(t - 0.5), 10), t))


def dev_threshold(student, task, records, labels, cfg):
    scores, predictions = {}, {}
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        predictions[threshold] = task.predict(
            student,
            records,
            labels,
            threshold=threshold,
            batch_size=cfg.batch_size,
            slot_mode=cfg.slot_mode,
        )
        scores[threshold] = task.score(records, predictions[threshold], labels)["micro"]["f1"]
    chosen = choose_eval_threshold(scores)
    return chosen, scores, predictions[chosen]


def finish_run(
    cfg, seeds, run_dir, task, labels, training, evaluation, n_selected, started, metric_labels=None
) -> Path:
    if cfg.finetune == "full":
        # A full fine-tune saves the whole model, not an adapter.
        student = model.AutoExtractor.from_pretrained(
            str(run_dir / "adapter/best"), map_location="cuda"
        )
    else:
        student = model.load_adapter(model.load_student(device="cuda"), run_dir / "adapter/best")
    student.eval()
    (run_dir / "predictions").mkdir()
    (run_dir / "plots").mkdir()
    metrics = {
        **training,
        "seeds": seeds,
        "n_selected": n_selected,
        **({"metric_labels": metric_labels} if metric_labels else {}),
    }
    for filename in ("acquisition.json", "protocol.json"):
        if (run_dir / filename).exists():
            metrics.update(json.loads((run_dir / filename).read_text()))
    chosen, dev_scores, dev_predictions = dev_threshold(
        student, task, evaluation["dev"], labels, cfg
    )
    metrics["eval_threshold_dev_chosen"] = chosen
    metrics["eval_threshold_dev_scores"] = dev_scores
    for split in evaluation:
        predictions = (
            dev_predictions
            if split == "dev"
            else task.predict(
                student,
                evaluation[split],
                labels,
                threshold=chosen,
                batch_size=cfg.batch_size,
                slot_mode=cfg.slot_mode,
            )
        )
        write_jsonl(run_dir / f"predictions/{split}.jsonl", predictions)
        metrics[split] = task.score(evaluation[split], predictions, labels)
    if "test" in evaluation:
        sweep = []
        for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
            descriptive = (
                predictions
                if threshold == chosen and split == "test"
                else task.predict(
                    student,
                    evaluation["test"],
                    labels,
                    threshold=threshold,
                    batch_size=cfg.batch_size,
                    slot_mode=cfg.slot_mode,
                )
            )
            sweep.append(
                {
                    "purpose": "descriptive_only_not_for_choices",
                    "split": "test",
                    "threshold": threshold,
                    "micro_f1": task.score(evaluation["test"], descriptive, labels)["micro"]["f1"],
                }
            )
        write_csv(
            run_dir / "threshold_sweep.csv", sweep, ["purpose", "split", "threshold", "micro_f1"]
        )
    per_type = per_type_rows(task, metrics[split], predictions)
    write_csv(
        run_dir / "per_type.csv",
        per_type,
        ["label", "precision", "recall", "f1", "tp", "fp", "fn", "support", "mean_confidence"],
    )
    calibration = calibration_rows(task, evaluation[split], predictions)
    write_csv(
        run_dir / "calibration.csv", calibration, ["bin", "count", "mean_confidence", "correctness"]
    )
    write_jsonl(run_dir / "errors.jsonl", error_rows(task, evaluation[split], predictions))
    plot_training(run_dir, metrics["best_step"], metrics["total_steps"])
    plot_calibration(run_dir, calibration)
    metrics["wall_time_s"] = time.monotonic() - started
    write_json(run_dir / "metrics.json", metrics)
    write_report(run_dir, cfg, metrics, per_type)
    return run_dir
