"""LoRA training with independent seeds and live logs."""

import shutil
import time
from dataclasses import fields
from pathlib import Path

import torch
from gliner2.processor import SamplingConfig
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

from active_gliner.observe.logs import append_json, write_json
from active_gliner.observe.resources import sample


class RunTrainer(ExtractorTrainer):
    def __init__(self, *args, run_dir: Path, lora_seed: int, eval_callback=None, **kwargs):
        self.eval_callback = eval_callback
        self.run_dir = run_dir
        self.lora_seed = lora_seed
        self.started = time.monotonic()
        self.stopped_early = False
        super().__init__(*args, **kwargs)

    def _setup_lora(self):
        from gliner2.training.lora_targets import _resolve_targets

        targets = self.config.lora_target_modules
        if set(targets) == {"encoder", "all_task_heads"}:
            count = len(_resolve_targets(self.model, targets))
            if count != 131:
                raise ValueError(f"Expected 131 student LoRA targets, found {count}")
        # The parent seeds data order before this hook. Isolate adapter initialisation.
        with torch.random.fork_rng():
            torch.manual_seed(self.lora_seed)
            super()._setup_lora()

    def _check_early_stopping(self, metrics, prev_best=None):
        self.stopped_early = super()._check_early_stopping(metrics, prev_best)
        return self.stopped_early

    def _log_metrics(self, metrics, prefix=""):
        row = dict(metrics) if isinstance(metrics, dict) else metrics.to_dict()
        row["step"] = self.global_step
        row["time_s"] = time.monotonic() - self.started
        if prefix == "eval":
            append_json(self.run_dir / "eval_log.jsonl", row)
            if self.eval_callback is not None:
                self.eval_callback(self.global_step, row["dev_f1"])
        elif "loss" in row:
            row.update(sample())
            append_json(self.run_dir / "train_log.jsonl", row)
        super()._log_metrics(metrics, prefix)


def exact_numerics_settings(exact: bool) -> dict:
    """TrainingConfig fields for bit-repeatable training; empty keeps gliner2 defaults."""
    if not exact:
        return {}
    return {
        "deterministic": True,
        "fp16": False,
        "bf16": False,
        "allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def apply_exact_numerics(exact: bool) -> None:
    """Make every later GPU op in this process bit-repeatable, or fail loudly.

    warn_only=True is not enough: memory-efficient attention keeps its
    non-deterministic backward and only warns, so same-seed runs drifted apart.
    """
    if not exact:
        return
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def train(
    student, cfg, seeds: dict, run_dir: Path, task, labels: dict, selected, dev, eval_callback=None
) -> dict:
    from active_gliner.exact_targets import install_exact_targets
    from active_gliner.tasks._spans import training_targets_match

    if selected[0].task == "relations" and cfg.relation_head != "classifier":
        from active_gliner.tasks.relation_training import train_relations

        apply_exact_numerics(cfg.exact_numerics)

        return train_relations(student, cfg, seeds, run_dir, labels, selected, dev, eval_callback)
    if selected[0].task in {"ner", "slots"}:
        install_exact_targets(student)
    if not cfg.augmentation or selected[0].task == "relations":
        sampling = SamplingConfig()
        for field in fields(sampling):
            if field.name.endswith("_prob"):
                setattr(sampling, field.name, 0.0)
            elif field.name.startswith("shuffle_"):
                setattr(sampling, field.name, False)
        student.processor.sampling_config = sampling

    def examples(records):
        kwargs = {"slot_mode": cfg.slot_mode} if records[0].task == "slots" else {}
        result = []
        for record in records:
            example = task.to_training_example(record, labels, **kwargs)
            if record.task in {"ner", "slots"} and not training_targets_match(record, example):
                raise ValueError(f"Training targets differ from supplied spans: {record.id}")
            result.extend(example if isinstance(example, list) else [example])
        return result

    def compute_metrics(current, _dataset):
        predictions = task.predict(
            current,
            dev,
            labels,
            threshold=cfg.eval_threshold,
            batch_size=cfg.batch_size,
            slot_mode=cfg.slot_mode,
        )
        return {"dev_f1": task.score(dev, predictions, labels)["micro"]["f1"]}

    train_examples = examples(selected)
    pair_counts = {}
    if selected[0].task == "relations":
        from active_gliner.tasks.relations import subsample_negatives

        train_examples, pair_counts = subsample_negatives(
            train_examples, cfg.relation_neg_ratio, seeds["data_order"]
        )
        write_json(run_dir / "training_pairs.json", pair_counts)
        if not train_examples:
            raise ValueError("Relation subsampling leaves no training pairs: no positive pairs")

    config = TrainingConfig(
        output_dir=str(run_dir / "trainer"),
        use_lora=True,
        max_steps=cfg.max_steps,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        task_lr=cfg.task_lr,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        metric_for_best="dev_f1",
        greater_is_better=True,
        early_stopping=True,
        early_stopping_patience=cfg.early_stopping_patience,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_targets,
        seed=seeds["data_order"],
        save_total_limit=1,
        num_workers=0,
        logging_steps=1,
        **exact_numerics_settings(cfg.exact_numerics),
    )
    apply_exact_numerics(cfg.exact_numerics)
    trainer = RunTrainer(
        student,
        config,
        run_dir=run_dir,
        lora_seed=seeds["lora_init"],
        compute_metrics=compute_metrics,
        eval_callback=eval_callback,
    )
    started = time.monotonic()
    result = trainer.train(train_data=train_examples, eval_data=examples(dev))
    # The package does not evaluate a final step outside the evaluation interval.
    if (
        not trainer.eval_metrics_history
        or trainer.eval_metrics_history[-1]["step"] != trainer.global_step
    ):
        trainer._evaluate(trainer._prepare_data(examples(dev), is_train=False))
    elapsed = time.monotonic() - started
    best = max(trainer.eval_metrics_history, key=lambda row: row["dev_f1"])
    shutil.copytree(run_dir / "trainer/best", run_dir / "adapter/best")
    shutil.copy2(run_dir / "trainer/training_config.json", run_dir / "training_config.json")
    shutil.rmtree(run_dir / "trainer")
    return {
        **({"training_pairs": pair_counts} if pair_counts else {}),
        "best_step": best["step"],
        "best_dev_f1": best["dev_f1"],
        "stop_reason": "early_stopping" if trainer.stopped_early else "max_steps",
        "train_time_s": elapsed,
        "total_steps": result["total_steps"],
    }
