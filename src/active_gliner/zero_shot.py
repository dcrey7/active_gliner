"""Zero-shot reference ladder (design section 13): inference only, no training.

Each model scores dev with the frozen threshold rule, then test once at the chosen
threshold. The run records latency and peak GPU memory for the Pareto figure.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from active_gliner import data, model, pins, protocol, tasks
from active_gliner.observe.logs import write_json, write_jsonl
from active_gliner.run import RunConfig, dev_threshold

# Pinned to each repo's main snapshot on 26 Sep 2026; multi matches the student pin.
LADDER = {
    "gliner2.5-small-v1": (
        "fastino/gliner2.5-small-v1",
        "3ec6d3dd7e1e93a7cf9b46096fa47aeda61c711c",
    ),
    "gliner2.5-base-v1": ("fastino/gliner2.5-base-v1", "b0c10b23313ec3ff028821dff298dd743e010706"),
    "gliner2.5-multi-v1": (model.STUDENT_REPO, model.STUDENT_SHA),
    "gliner2-large-v1": ("fastino/gliner2-large-v1", "bf90d758a5d482bbfc276041b8cb7b570e5318e3"),
}
# The six datasets plus MASSIVE French: 7 pairs per model.
PAIRS = [
    ("cleanconll", "en-US"),
    ("bc5cdr", "en-US"),
    ("mit_movie", "en-US"),
    ("crossre", "en-US"),
    ("hallmarks", "en-US"),
    ("massive", "en-US"),
    ("massive", "fr-FR"),
]
# One batch size for every model keeps the latency comparison fair.
BATCH_SIZE = 8
# Marker file for a pair the model cannot score (no metrics are written).
UNSUPPORTED = "UNSUPPORTED.txt"


def run_dir(name: str, dataset: str, locale: str, out_root="runs") -> Path:
    return Path(out_root) / "zero_shot" / name / dataset / locale


def evaluate(name: str, dataset: str, locale: str = "en-US", out_root="runs") -> Path | None:
    """Score one ladder model on one dataset; skip it when its metrics already exist.

    Returns None when the model cannot score the task (see UNSUPPORTED).
    """
    import torch

    repo, sha = LADDER[name]
    directory = run_dir(name, dataset, locale, out_root)
    if (directory / "metrics.json").exists():
        return directory
    if (directory / UNSUPPORTED).exists():
        return None
    task = tasks.for_dataset(dataset)
    splits = data.load(dataset, locale)
    labels = tasks.labels(dataset, splits)
    task_name = splits["dev"][0].task
    defaults = RunConfig.model_fields
    cfg = SimpleNamespace(batch_size=BATCH_SIZE, slot_mode=defaults["slot_mode"].default)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "zero_shot",
                "dataset": dataset,
                "locale": locale,
                "labels_source": "none",
                "selector": "zero_shot",
                "n": 0,
                "seed": 0,
                "model": name,
                "batch_size": BATCH_SIZE,
                "slot_mode": cfg.slot_mode,
            }
        ),
        encoding="utf-8",
    )
    pins.write_pins(
        directory / "pins.json",
        pins.collect_pins(repo, sha, {dataset: data.dataset_pin(dataset)}),
    )

    student = model.load_student(repo, sha, device="cuda")
    if task_name == "relations":
        from active_gliner.tasks.relation_heads import configure

        # CrossRE is scored with the pretrained native relation scorer (design section 17).
        try:
            configure(student, "native", labels)
        except ValueError as exc:
            # GLiNER2 large has no relation scorer: record the gap instead of a fake score.
            (directory / UNSUPPORTED).write_text(f"{exc}\n", encoding="utf-8")
            del student
            model.free_gpu()
            return None
    if not hasattr(student, "boundary_settings"):
        # GLiNER2 large is an older SpanExtractor. Slot prediction in structure mode only
        # reads and restores this value, so None leaves its predictions unchanged.
        student.boundary_settings = None
    student.eval()
    torch.cuda.reset_peak_memory_stats()
    with protocol.inference_precision():
        chosen, dev_scores, dev_predictions = dev_threshold(
            student, task, splits["dev"], labels, cfg
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        test_predictions = task.predict(
            student,
            splits["test"],
            labels,
            threshold=chosen,
            batch_size=BATCH_SIZE,
            slot_mode=cfg.slot_mode,
        )
        torch.cuda.synchronize()
        test_seconds = time.perf_counter() - started

    (directory / "predictions").mkdir(exist_ok=True)
    write_jsonl(directory / "predictions/dev.jsonl", dev_predictions)
    write_jsonl(directory / "predictions/test.jsonl", test_predictions)
    metrics = {
        "model": repo,
        "zero_shot_name": name,
        "params": sum(p.numel() for p in student.parameters()),
        "eval_threshold_dev_chosen": chosen,
        "eval_threshold_dev_scores": dev_scores,
        "dev": task.score(splits["dev"], dev_predictions, labels),
        "test": task.score(splits["test"], test_predictions, labels),
        "latency_ms": 1000 * test_seconds / len(splits["test"]),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_sentences": len(splits["test"]),
        "timing_protocol": f"test predict, batch {BATCH_SIZE}, fp32, no TF32, synchronized",
    }
    write_json(directory / "metrics.json", metrics)
    del student
    model.free_gpu()
    return directory


def evaluate_all(names=None, out_root="runs") -> list[dict]:
    rows = []
    for name in names or LADDER:
        for dataset, locale in PAIRS:
            directory = evaluate(name, dataset, locale, out_root)
            if directory is None:
                rows.append(
                    {"model": name, "dataset": dataset, "locale": locale, "unsupported": True}
                )
                continue
            metrics = json.loads((directory / "metrics.json").read_text())
            rows.append(
                {
                    "model": name,
                    "dataset": dataset,
                    "locale": locale,
                    "test_f1": metrics["test"]["micro"]["f1"],
                    "threshold": metrics["eval_threshold_dev_chosen"],
                }
            )
    return rows
