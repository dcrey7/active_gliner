"""Frozen, document-grouped test samples for the English teacher ladder."""

import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles

import numpy as np

from active_gliner.teachers.config import REPO_ROOT

SEED = 20260925


def sample(
    items: list[tuple[str, str | None, float]],
    target: int = 1000,
    n_strata: int = 4,
    seed: int = 0,
) -> list[dict]:
    if target <= 0 or n_strata <= 0:
        raise ValueError("target and n_strata must be positive")
    if len({id for id, _, _ in items}) != len(items):
        raise ValueError("Sentence ids must be unique")
    if any(not math.isfinite(confidence) for _, _, confidence in items):
        raise ValueError("Confidence must be finite")
    if not items:
        return []
    boundaries = (
        quantiles([confidence for _, _, confidence in items], n=n_strata, method="inclusive")
        if len(items) > 1 and n_strata > 1
        else []
    )
    units = defaultdict(list)
    for id, doc_id, confidence in items:
        units[("sentence", id) if doc_id is None else ("document", doc_id)].append((id, confidence))
    strata = defaultdict(list)
    for unit, sentences in units.items():
        # Quantile boundaries use sentences; each document stays in one stratum.
        stratum = bisect_right(boundaries, mean(c for _, c in sentences))
        strata[stratum].append(unit)
    rng = random.Random(seed)
    selected = {}
    for stratum, members in sorted(strata.items()):
        mean_size = mean(len(units[unit]) for unit in members)
        count = (
            len(members)
            if len(items) <= target
            else min(len(members), math.ceil(target / n_strata / mean_size))
        )
        chosen = rng.sample(members, count)
        probability = len(chosen) / len(members)
        for unit in chosen:
            for id, _ in units[unit]:
                selected[id] = (stratum, probability)
    return [
        dict(id=id, doc_id=doc_id, stratum=selected[id][0], inclusion_prob=selected[id][1])
        for id, doc_id, _ in items
        if id in selected
    ]


def _sample_path(dataset: str, locale: str) -> Path:
    from active_gliner import tasks

    tasks.for_dataset(dataset)
    if locale != "en-US":
        raise ValueError("The teacher ladder is English-only")
    return REPO_ROOT / "data/ladder" / f"{dataset}-{locale}.json"


def sample_dataset(dataset: str, locale: str = "en-US") -> Path:
    from active_gliner import data, model, tasks

    path = _sample_path(dataset, locale)
    if path.exists():
        return path
    splits = data.load(dataset, locale)
    records = splits["test"]
    if not records:
        raise ValueError("The test split is empty")
    student = model.load_student()
    student.eval()
    predictions = tasks.for_dataset(dataset).predict(
        student, records, tasks.labels(dataset, splits), threshold=0.5
    )
    confidences = {p["id"]: p["confidence"] for p in predictions}
    rows = sample([(r.id, r.doc_id, confidences[r.id]) for r in records], seed=SEED)
    value = dict(dataset=dataset, locale=locale, seed=SEED, target=1000, rows=rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second invocation from replacing frozen ids.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return path


def label_sample(dataset: str, teacher: str, locale: str = "en-US") -> dict:
    from active_gliner import data, tasks
    from active_gliner.teachers import TeacherConfig, label_records, prompts

    value = json.loads(_sample_path(dataset, locale).read_text(encoding="utf-8"))
    if value["dataset"] != dataset or value["locale"] != locale:
        raise ValueError("Sample dataset or locale does not match")
    splits = data.load(dataset, locale)
    by_id = {r.id: r for r in splits["test"]}
    ids = [row["id"] for row in value["rows"]]
    if not ids or len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("Sample must contain unique ids from the test split")
    records = [by_id[id] for id in ids]
    cfg = TeacherConfig.from_yaml(REPO_ROOT / "configs/teachers" / f"{teacher}.yaml")
    version = prompts.version_for(dataset)
    result = label_records(
        records,
        records[0].task,
        tasks.labels(dataset, splits),
        cfg,
        REPO_ROOT / "labels",
        dataset=dataset,
        definitions=prompts.prompt_for(dataset, version),
        prompt_version=version,
    )
    return result.stats


def weighted_f1(counts, inclusion_prob) -> float:
    """Micro-F1 of the whole test split, estimated from a stratified sample.

    Each sentence counts 1 / inclusion probability times (Horvitz-Thompson), so strata
    that were sampled less often weigh more.
    """
    from .stats import micro_f1

    counts = np.asarray(counts, dtype=float)
    weights = 1 / np.asarray(inclusion_prob, dtype=float)
    return micro_f1(counts * weights[:, None])


def gap_interval(rows, counts_a, counts_b, n_boot=1000, seed=0) -> dict:
    """95% interval of F1(a) - F1(b), in points, resampling sample units within strata.

    A unit is a document when the sample kept whole documents, else a sentence. The
    interval covers sampling from the test split only, not teacher randomness.
    """
    units = defaultdict(list)
    for index, row in enumerate(rows):
        key = row["id"] if row["doc_id"] is None else row["doc_id"]
        units[(row["stratum"], key)].append(index)
    by_stratum = defaultdict(list)
    for stratum, key in units:
        by_stratum[stratum].append(units[(stratum, key)])
    probability = np.array([row["inclusion_prob"] for row in rows])
    counts_a, counts_b = np.asarray(counts_a), np.asarray(counts_b)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        index = []
        for members in by_stratum.values():
            for pick in rng.integers(len(members), size=len(members)):
                index.extend(members[pick])
        a = weighted_f1(counts_a[index], probability[index])
        b = weighted_f1(counts_b[index], probability[index])
        gaps.append(100 * (a - b))
    low, high = np.percentile(gaps, [2.5, 97.5])
    return dict(low=float(low), high=float(high), n_boot=n_boot)


def scores(dataset: str, teachers: list[str], locale: str = "en-US") -> dict:
    """Teacher F1 on the frozen ladder sample and the overlap of their error sentences.

    A teacher whose labels do not cover the sample is left out.
    """
    from . import teacher as teacher_diagnostics

    value = json.loads(_sample_path(dataset, locale).read_text(encoding="utf-8"))
    ids = [row["id"] for row in value["rows"]]
    probability = [row["inclusion_prob"] for row in value["rows"]]
    f1, errors, counts = {}, {}, {}
    for name in teachers:
        try:
            counts[name] = teacher_diagnostics.teacher_scores(dataset, locale, name, "test", ids)
        except FileNotFoundError:
            continue
        f1[name] = 100 * weighted_f1(counts[name], probability)
        errors[name] = {id for id, row in zip(ids, counts[name], strict=True) if row[1] or row[2]}
    result = dict(
        n=len(ids),
        documents=len({row["doc_id"] for row in value["rows"] if row["doc_id"] is not None}),
        f1=f1,
        error_overlap=teacher_diagnostics.error_overlap(errors).to_dict() if errors else {},
    )
    if len(teachers) == 2 and len(counts) == 2:
        first, second = teachers[0], teachers[1]
        result["gap"] = dict(
            mean=f1[first] - f1[second],
            interval=gap_interval(value["rows"], counts[first], counts[second]),
        )
    return result
