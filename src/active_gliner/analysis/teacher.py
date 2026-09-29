"""Teacher diagnostics on matched sentences, separate from the primary claims."""

from dataclasses import replace

import numpy as np
import pandas as pd

from .aggregate import item_counts
from .stats import _counts, micro_f1


def f1_by_bin(confidence, counts, n_bins=4) -> pd.DataFrame:
    confidence = np.asarray(confidence, dtype=float)
    counts = _counts(counts)
    if confidence.ndim != 1 or len(confidence) != len(counts) or not np.isfinite(confidence).all():
        raise ValueError("Confidence must be finite and match the sentence counts")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    rows = []
    # Stable ranks give disjoint, equally sized bins even when confidence has ties.
    for index, ids in enumerate(np.array_split(np.argsort(confidence, kind="stable"), n_bins)):
        rows.append(
            dict(
                bin=index,
                low=float(confidence[ids].min()) if len(ids) else None,
                high=float(confidence[ids].max()) if len(ids) else None,
                n=len(ids),
                f1=micro_f1(counts[ids]) if len(ids) else None,
            )
        )
    return pd.DataFrame(rows)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def error_overlap(teacher_error_sets: dict) -> pd.DataFrame:
    names = sorted(teacher_error_sets)
    return pd.DataFrame(
        [[jaccard(teacher_error_sets[a], teacher_error_sets[b]) for b in names] for a in names],
        index=names,
        columns=names,
    )


def cached_labels(dataset, locale, teacher, split, ids=None):
    """Cached teacher labels for a split, or for the given sentence ids of that split."""
    from active_gliner import data, tasks
    from active_gliner.teachers import prompts
    from active_gliner.teachers.cache import LabelCache
    from active_gliner.teachers.config import REPO_ROOT

    splits = data.load(dataset, locale)
    records = splits[split]
    labels = tasks.labels(dataset, splits)
    if ids is not None:
        by_id = {record.id: record for record in records}
        records = [by_id[id] for id in ids]
    task = tasks.for_dataset(dataset)
    digest = prompts.prompt_hash(records[0].task, labels, prompts.prompt_for(dataset))
    cache = LabelCache.for_job(
        REPO_ROOT / "labels", teacher, records[0].task, dataset, locale, digest
    )
    values = [cache.get(record.id) for record in records]
    missing = sum(value is None for value in values)
    if missing:
        raise FileNotFoundError(f"{missing} {split} labels missing in {cache.path}")
    return task, records, values


def teacher_scores(dataset, locale, teacher, split, ids=None) -> np.ndarray:
    """Counts in split order (or `ids` order); invalid output predicts no positive items."""
    task, records, values = cached_labels(dataset, locale, teacher, split, ids)
    return np.asarray(
        [
            item_counts(
                task.gold_items(record),
                task.gold_items(replace(record, gold=value["gold"]))
                if value["gold"] is not None
                else set(),
            )
            for record, value in zip(records, values, strict=True)
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
