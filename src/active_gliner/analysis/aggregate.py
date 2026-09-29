"""Read finished runs without loading a student model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TASKS = dict(
    cleanconll="ner",
    bc5cdr="ner",
    mit_movie="ner",
    crossre="relations",
    hallmarks="classification",
    massive="slots",
)
COLUMNS = (
    "name block dataset locale task labels selector n seed gt_fraction gt_assignment "
    "no_prediction dev_f1 f1 macro_f1 wall_time_s train_time_s teacher_prompt_tokens "
    "teacher_completion_tokens teacher_latency_s run_dir variant"
).split()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(runs_root="runs") -> pd.DataFrame:
    rows = []
    root = Path(runs_root)
    for path in sorted(root.rglob("metrics.json")):
        directory = path.parent
        if not (directory / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((directory / "config.yaml").read_text())
        metrics = read_json(path)
        if "test" not in metrics:
            continue
        row = {key: cfg.get(key) for key in COLUMNS}
        teacher_path = directory / "teacher_stats.json"
        teacher = read_json(teacher_path) if teacher_path.exists() else {}
        pins_path = directory / "pins.json"
        if pins_path.exists():
            row["model"] = read_json(pins_path).get("model", {}).get("repo")
        row.update(
            name=str(directory.relative_to(root)),
            run_dir=str(directory),
            task=cfg.get("task", TASKS.get(cfg["dataset"])),
            locale=cfg.get("locale", "en-US"),
            labels=cfg["labels_source"],
            n=metrics.get("n_selected", cfg.get("n")),
            no_prediction=cfg.get("no_prediction", "zero"),
            dev_f1=metrics.get("dev", {}).get("micro", {}).get("f1"),
            f1=100 * metrics["test"]["micro"]["f1"],
            macro_f1=100 * metrics["test"].get("macro_f1", float("nan")),
        )
        for key in ("wall_time_s", "train_time_s"):
            row[key] = metrics.get(key)
        # Thesis E6: how much dev F1 moves across the candidate thresholds (points).
        sweep = metrics.get("eval_threshold_dev_scores") or {}
        row["dev_threshold_spread"] = (
            100 * (max(sweep.values()) - min(sweep.values())) if sweep else None
        )
        for field, source in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("latency_s", "latency_s_total"),
        ):
            row[f"teacher_{field}"] = teacher.get(source)
        # Optional measurements stay missing rather than becoming free compute.
        for key in (
            "teacher_gpu_h",
            "scoring_gpu_h",
            "train_gpu_h",
            "inference_cost_1m",
            "latency_ms",
            "params",
            "family",
            "zero_shot_name",
            "coverage",
            "timing_protocol",
            "cost_usd",
            "model",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(dict.fromkeys(COLUMNS + [k for r in rows for k in r])))


def item_counts(gold: set, predicted: set) -> tuple[int, int, int]:
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def test_ids(run_dir) -> list[str]:
    path = Path(run_dir) / "predictions/test.jsonl"
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate test predictions: {path}")
    return sorted(ids)


def sentence_counts(run_dir) -> np.ndarray:
    """Return counts in sorted sentence-id order, shared by all paired arms."""
    directory = Path(run_dir)
    cache = directory / "test_counts.npy"
    inputs = [directory / "config.yaml", directory / "predictions/test.jsonl"]
    inputs += [
        directory / name for name in ("hashes.json", "pins.json") if (directory / name).exists()
    ]
    if cache.exists() and cache.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in inputs):
        from .stats import _counts

        counts = _counts(np.load(cache, allow_pickle=False))
        if len(counts) != len(test_ids(directory)):
            raise ValueError(f"Cached counts do not match prediction IDs: {directory}")
        return counts

    from active_gliner import data, tasks

    cfg = yaml.safe_load((directory / "config.yaml").read_text())
    splits = data.load(cfg["dataset"], cfg.get("locale", "en-US"))
    records = splits["test"][: cfg.get("test_limit")]
    expected = {record.id: record for record in records}
    predictions = [
        json.loads(line)
        for line in (directory / "predictions/test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    actual = {p["id"]: p for p in predictions}
    if len(actual) != len(predictions) or set(actual) != set(expected):
        raise ValueError(f"Test prediction IDs do not match the test split: {directory}")
    hashes = directory / "hashes.json"
    if hashes.exists() and "split" in read_json(hashes):
        limited = {key: values[: cfg.get(f"{key}_limit")] for key, values in splits.items()}
        if data.split_hash(limited) != read_json(hashes)["split"]:
            raise ValueError(f"Frozen split hash differs: {directory}")
    task = tasks.for_dataset(cfg["dataset"])
    counts = np.asarray(
        [
            item_counts(task.gold_items(expected[id]), task.pred_items(actual[id]))
            for id in sorted(expected)
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    np.save(directory / "test_counts.npy", counts)
    return counts
