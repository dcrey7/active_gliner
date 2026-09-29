"""Paired effects in F1 points; bootstrap intervals cover test uncertainty only."""

from pathlib import Path

import numpy as np
import pandas as pd

TEACHER = "gemma-4-12b"


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def _counts(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 2
        or counts.shape[1] != 3
        or not np.isfinite(counts).all()
        or (counts < 0).any()
    ):
        raise ValueError("Counts must be a finite nonnegative (sentences, 3) array")
    return counts


def micro_f1(counts) -> float:
    tp, fp, fn = _counts(counts).sum(axis=0)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _interval(diff, samples) -> dict:
    samples = np.asarray(samples)
    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(
        diff=float(diff),
        low=float(low),
        high=float(high),
        p_value=float(min(1, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))),
    )


def bootstrap_diff(counts_b, counts_a, n_boot=1000, seed=0) -> dict:
    a, b = _counts(counts_a), _counts(counts_b)
    if a.shape != b.shape or not len(a) or n_boot < 1:
        raise ValueError("Paired bootstrap needs equal nonempty arrays and positive n_boot")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        indices = rng.integers(len(a), size=len(a))
        samples.append(100 * (micro_f1(b[indices]) - micro_f1(a[indices])))
    return _interval(100 * (micro_f1(b) - micro_f1(a)), samples)


def _cells(df, teacher, n=None):
    if n is not None:
        df = df[df.n == n]
    datasets = sorted(df.dataset.unique())
    columns = [
        (teacher, "min"),
        (teacher, "random"),
        ("ground_truth", "min"),
        ("ground_truth", "random"),
    ]
    df = df[df.labels.isin([teacher, "ground_truth"]) & df.selector.isin(["min", "random"])]
    keys = ["dataset", "seed", "labels", "selector"]
    if df.duplicated(keys).any():
        raise ValueError("Ambiguous cells: filter budget, locale, mixing and sensitivity first")
    if not datasets:
        raise ValueError("No datasets for the interaction")
    table = df.pivot(index=["dataset", "seed"], columns=["labels", "selector"], values="f1")
    table = table.reindex(columns=pd.MultiIndex.from_tuples(columns)).dropna()
    seeds = set(df.seed.unique())
    for dataset in datasets:
        seeds &= {s for ds, s in table.index if ds == dataset}
    if not seeds:
        raise ValueError("No seed has all four cells in every dataset")
    seeds = sorted(seeds)
    index = pd.MultiIndex.from_product([datasets, seeds], names=["dataset", "seed"])
    return table.reindex(index), datasets, seeds


def _sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def interaction(df, teacher, n=None) -> dict:
    table, datasets, seeds = _cells(df, teacher, n)
    values = table.to_numpy().reshape(len(datasets), len(seeds), 4)
    practical = values[:, :, 0] - values[:, :, 1]
    effects = practical - (values[:, :, 2] - values[:, :, 3])
    pooled = effects.mean(axis=0)
    return dict(
        per_seed=pooled.tolist(),
        seeds=seeds,
        per_dataset=dict(zip(datasets, effects.mean(axis=1).tolist(), strict=True)),
        mean=float(effects.mean()),
        practical_mean=float(practical.mean()),
        practical_per_seed=practical.mean(axis=0).tolist(),
        sd=_sd(pooled),
        practical_sd=_sd(practical.mean(axis=0)),
    )


def interaction_bootstrap(run_rows, n_boot=1000, seed=0) -> dict:
    """Rows contain dataset, seed, labels, selector and run_dir (or aligned counts)."""
    from .aggregate import sentence_counts, test_ids

    rows = pd.DataFrame(run_rows).copy()
    teachers = set(rows.labels) - {"ground_truth"}
    if len(teachers) != 1 or n_boot < 1:
        raise ValueError("Bootstrap needs one teacher and a positive n_boot")
    teacher = teachers.pop()
    if "f1" not in rows:
        rows["f1"] = 0.0
    _, datasets, seeds = _cells(rows, teacher)
    rows = rows[rows.seed.isin(seeds) & rows.selector.isin(["min", "random"])]
    cells, ids_by_dataset = {}, {}
    for row in rows.to_dict("records"):
        if "counts" in row:
            counts = _counts(row["counts"])
            ids = row.get("ids", list(range(len(counts))))
        else:
            counts = sentence_counts(row["run_dir"])
            ids = test_ids(row["run_dir"])
        ds = row["dataset"]
        if ds in ids_by_dataset and ids_by_dataset[ds] != list(ids):
            raise ValueError(f"Paired test IDs differ for {ds}")
        ids_by_dataset[ds] = list(ids)
        if not len(counts) or len(ids) != len(counts):
            raise ValueError("Test counts must be nonempty and match IDs")
        cells[(ds, row["seed"], row["labels"], row["selector"])] = counts
    rng = np.random.default_rng(seed)

    def effects(indices):
        contrasts, practical = [], []
        for ds in datasets:
            for run_seed in seeds:
                scores = [
                    100 * micro_f1(cells[(ds, run_seed, label, selector)][indices[ds]])
                    for label, selector in [
                        (teacher, "min"),
                        (teacher, "random"),
                        ("ground_truth", "min"),
                        ("ground_truth", "random"),
                    ]
                ]
                practical.append(scores[0] - scores[1])
                contrasts.append(scores[0] - scores[1] - scores[2] + scores[3])
        return float(np.mean(contrasts)), float(np.mean(practical))

    estimate = effects({ds: np.arange(len(ids_by_dataset[ds])) for ds in datasets})
    samples = [
        effects({ds: rng.integers(len(ids), size=len(ids)) for ds, ids in ids_by_dataset.items()})
        for _ in range(n_boot)
    ]
    samples = np.asarray(samples)
    return {
        **_interval(estimate[0], samples[:, 0]),
        "practical": _interval(estimate[1], samples[:, 1]),
        "uncertainty": "test uncertainty only",
        "n_boot": n_boot,
        "seed": seed,
    }


def aso(scores_a, scores_b, seed) -> float:
    """Exploratory violation bound for A dominating B, never a p-value."""
    from deepsig import aso as deepsig_aso

    return float(
        deepsig_aso(
            scores_a,
            scores_b,
            seed=seed,
            confidence_level=0.95,
            num_bootstrap_iterations=1000,
            num_jobs=1,
            show_progress=False,
        )
    )


def missing_primary_cells(df, datasets, teacher=TEACHER) -> list[dict]:
    present = {
        (row.dataset, row.seed, row.labels, row.selector)
        for row in df.itertuples()
        if np.isfinite(row.f1)
    }
    return [
        dict(dataset=dataset, seed=seed, labels=label, selector=selector)
        for dataset in datasets
        for seed in range(1, 6)
        for label in (teacher, "ground_truth")
        for selector in ("min", "random")
        if (dataset, seed, label, selector) not in present
    ]


def primary_report(df, runs="runs") -> dict:
    specifications = {
        "ner": (["cleanconll", "bc5cdr", "mit_movie"], 400),
        "relations": (["crossre"], 200),
        "classification": (["hallmarks"], 400),
        "slots": (["massive"], 400),
    }
    report = {
        "interactions": {},
        "practical": {},
        "uncertainty": "test uncertainty only",
        "aso": {
            "exploratory": True,
            "confidence_level": 0.95,
            "n_boot": 1000,
            "seed": 0,
            "direction": {
                "interactions": "teacher min-random gap dominates gold min-random gap",
                "practical": "teacher min dominates teacher random",
            },
        },
    }
    for task, (datasets, budget) in specifications.items():
        subset = df[df.dataset.isin(datasets) & (df.n == budget)].copy()
        for key, value in (("locale", "en-US"), ("no_prediction", "zero")):
            if key in subset:
                subset = subset[subset[key].fillna(value) == value]
        if "gt_fraction" in subset:
            subset = subset[subset.gt_fraction.isna()]
        subset = subset[
            subset.labels.isin([TEACHER, "ground_truth"]) & subset.selector.isin(["min", "random"])
        ]
        subset = subset[subset.seed.isin(range(1, 6))]
        missing = [
            {**cell, "n": budget, "locale": "en-US"}
            for cell in missing_primary_cells(subset, datasets)
        ]
        if missing:
            for family in ("interactions", "practical"):
                report[family][task] = {"status": "incomplete", "missing_cells": missing}
            continue
        try:
            effect = interaction(subset, TEACHER)
        except ValueError as exc:
            for family in ("interactions", "practical"):
                report[family][task] = {"status": "unavailable", "reason": str(exc)}
            continue
        if "run_dir" not in subset:
            subset["run_dir"] = [str(Path(runs) / name) for name in subset.name]
        try:
            interval = interaction_bootstrap(subset, n_boot=1000, seed=0)
        except (FileNotFoundError, ValueError) as exc:
            interval = {"reason": str(exc), "practical": {"reason": str(exc)}}
        table, datasets_used, seeds_used = _cells(subset, TEACHER)
        pooled = table.to_numpy().reshape(len(datasets_used), len(seeds_used), 4).mean(axis=0)
        aso_pairs = {
            "interactions": (pooled[:, 0] - pooled[:, 1], pooled[:, 2] - pooled[:, 3]),
            "practical": (pooled[:, 0], pooled[:, 1]),
        }
        for family, values, mean, sd, ci in (
            ("interactions", effect["per_seed"], effect["mean"], effect["sd"], interval),
            (
                "practical",
                effect["practical_per_seed"],
                effect["practical_mean"],
                effect["practical_sd"],
                interval["practical"],
            ),
        ):
            entry = dict(
                status="available",
                mean=mean,
                sd=sd,
                per_seed=values,
                seeds=effect["seeds"],
                interval={k: v for k, v in ci.items() if k != "practical"},
            )
            if family == "interactions":
                entry["per_dataset"] = effect["per_dataset"]
            if len(values) > 1:
                try:
                    entry["epsilon_min"] = aso(*aso_pairs[family], seed=0)
                except ImportError:
                    entry["epsilon_min"] = None
                    entry["aso_unavailable"] = "deepsig is not installed"
            else:
                entry["epsilon_min"] = None
            report[family][task] = entry
    # Do not publish a partial Holm family.
    for family in ("interactions", "practical"):
        entries = list(report[family].values())
        if any(entry["status"] != "available" for entry in entries):
            continue
        adjusted = holm([entry.get("interval", {}).get("p_value", 1.0) for entry in entries])
        for entry, p in zip(entries, adjusted, strict=True):
            if entry["status"] == "available":
                entry["p_holm"] = p if "p_value" in entry.get("interval", {}) else None
    return report
