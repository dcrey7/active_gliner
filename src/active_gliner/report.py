"""A report card built from measured run output."""

from pathlib import Path


def write_report(run_dir: Path, cfg, metrics: dict, per_type: list[dict]) -> None:
    lines = [
        f"# {cfg.dataset}: {cfg.selector}, N={cfg.n}, seed={cfg.seed}",
        "",
        f"Locale: {cfg.locale}. Labels: {cfg.labels_source}. LoRA r={cfg.lora_r}, "
        f"alpha={cfg.lora_alpha}, task LR={cfg.task_lr}. Augmentation: {cfg.augmentation}.",
        f"Acquisition threshold: {cfg.acq_threshold}. Evaluation threshold: {cfg.eval_threshold}.",
        "",
        "| Split | Micro F1 | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for split in ("dev", "test"):
        if split not in metrics:
            continue
        score = metrics[split]
        name = metrics.get("metric_labels", {}).get(split, split)
        lines.append(f"| {name} | {score['micro']['f1']:.4f} | {score['macro_f1']:.4f} |")
    lines += [
        "",
        f"Best step: {metrics['best_step']}. Stop reason: {metrics['stop_reason']}.",
        f"Wall time: {metrics['wall_time_s']:.2f} s. "
        f"Training time: {metrics['train_time_s']:.2f} s.",
        "",
        "| Type | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in per_type:
        lines.append(
            f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} |"
        )
    lines += [
        "",
        "[Training curves](plots/training_curves.png) · "
        "[Calibration](plots/calibration.png) · [Errors](errors.jsonl)",
        "",
        "Extraction calibration includes emitted predictions only; low bins can be empty "
        "because of the evaluation threshold. Classification includes all label probabilities.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
