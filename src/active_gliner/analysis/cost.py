"""Measured compute and estimated USD; missing measurements are never zero."""

import math
from pathlib import Path

import pandas as pd
import yaml


def prices() -> dict:
    from active_gliner.teachers.config import REPO_ROOT

    return yaml.safe_load((REPO_ROOT / "configs/prices.yaml").read_text())


def _value(row, key):
    value = row.get(key)
    return float(value) if value is not None and pd.notna(value) else None


def run_cost(row) -> dict:
    from .aggregate import read_json

    row = dict(row)
    directory = row.get("run_dir")
    if directory:
        path = Path(directory) / "metrics.json"
        if path.exists():
            row = {**read_json(path), **row}
        stats_path = Path(directory) / "teacher_stats.json"
        if stats_path.exists():
            stats = read_json(stats_path)
            row.update(
                teacher_elapsed_s=stats.get("elapsed_s"),
                request_latency_s=stats.get("latency_s_total"),
                gpu_shared=stats.get("gpu_shared", False),
            )
        if stats_path.exists() and read_json(stats_path).get("cache_hits", 0):
            # Job totals omit cached calls. They cannot price the full annotation set.
            for field in (
                "teacher_prompt_tokens",
                "teacher_completion_tokens",
                "teacher_latency_s",
                "teacher_elapsed_s",
            ):
                row[field] = None
    table = prices()
    label = row.get("labels", row.get("labels_source"))
    teacher = table["teachers"].get(label)
    gpu = _value(row, "teacher_gpu_h")
    usd = None
    if label == "ground_truth":
        gpu, usd = 0.0, 0.0
    elif teacher:
        if teacher["provider"] == "local":
            if _value(row, "teacher_elapsed_s") is not None:
                # Dedicated server allocation time; concurrent request durations overlap.
                gpu = _value(row, "teacher_elapsed_s") / 3600
            usd = gpu * table["gpu"]["usd_per_hour"] if gpu is not None else None
        else:
            gpu = 0.0
            prompt, completion = (
                _value(row, f"teacher_{kind}_tokens") for kind in ("prompt", "completion")
            )
            if prompt is not None and completion is not None:
                usd = (
                    prompt * teacher["prompt_usd_per_million"]
                    + completion * teacher["completion_usd_per_million"]
                ) / 1e6
    scoring = _value(row, "scoring_gpu_h")
    if row.get("selector") in {"random", "all"}:
        scoring = 0.0
    train = _value(row, "train_gpu_h")
    if train is None and _value(row, "train_time_s") is not None:
        # The run executor trains on one CUDA device for the logged duration.
        train = _value(row, "train_time_s") / 3600
    return dict(
        request_latency_s=_value(row, "request_latency_s")
        if "request_latency_s" in row
        else _value(row, "teacher_latency_s"),
        gpu_allocation_note="shared GPU; elapsed time is not exclusive GPU use"
        if row.get("gpu_shared")
        else "dedicated local server assumed",
        teacher_usd_estimated=usd,
        teacher_gpu_h=gpu,
        scoring_gpu_h=scoring,
        train_gpu_h=train,
    )


def equal_cost_table(df, dataset) -> pd.DataFrame:
    """Use the largest measured N within each budget, independently for each seed."""
    rows = df[df.dataset == dataset].copy()
    columns = [
        "budget",
        "budget_usd_estimated",
        "locale",
        "labels",
        "selector",
        "gt_fraction",
        "gt_assignment",
        "no_prediction",
        "seed",
        "n",
        "f1",
        "cost_usd_estimated",
        "status",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    # Gold labels need a stated human rate; do not treat them as free annotation.
    for key, default in (
        ("locale", "en-US"),
        ("gt_fraction", 0.0),
        ("gt_assignment", "none"),
        ("no_prediction", "zero"),
    ):
        if key not in rows:
            rows[key] = default
        rows[key] = rows[key].fillna(default)
    rate = prices()["gpu"]["usd_per_hour"]
    costs = []
    for row in rows.to_dict("records"):
        measured = run_cost(row)
        annotation, scoring = measured["teacher_usd_estimated"], measured["scoring_gpu_h"]
        costs.append(
            annotation + rate * scoring
            if annotation is not None
            and scoring is not None
            and not row["gt_fraction"]
            and row["labels"] != "ground_truth"
            else float("nan")
        )
    rows["acquisition_cost"] = costs
    output = []
    methods = ["labels", "selector", "gt_fraction", "gt_assignment", "no_prediction"]
    for locale, local in rows.groupby("locale"):
        baseline = local[
            (local.labels == "gemma-4-12b")
            & (local.gt_fraction == 0)
            & (local.no_prediction == "zero")
        ]
        budgets = {
            "N100": baseline[(baseline.selector == "random") & (baseline.n == 100)],
            "N400": baseline[(baseline.selector == "random") & (baseline.n == 400)],
            "full_pool": baseline[baseline.selector == "all"],
        }
        for budget_name, reference in budgets.items():
            budget = (
                reference.acquisition_cost.mean()
                if reference.acquisition_cost.notna().all()
                else float("nan")
            )
            for keys, group in local.groupby(methods + ["seed"], dropna=False):
                result = dict(zip(methods + ["seed"], keys, strict=True))
                result.update(
                    budget=budget_name,
                    locale=locale,
                    budget_usd_estimated=budget,
                    status="unavailable",
                )
                eligible = (
                    group[group.acquisition_cost <= budget]
                    if math.isfinite(budget)
                    else group.iloc[:0]
                )
                if not eligible.empty:
                    largest = eligible[eligible.n == eligible.n.max()]
                    if len(largest) != 1:
                        raise ValueError("Duplicate method, budget and seed in cost comparison")
                    chosen = largest.iloc[0]
                    result.update(
                        n=int(chosen.n),
                        f1=float(chosen.f1),
                        cost_usd_estimated=float(chosen.acquisition_cost),
                        status="available",
                    )
                output.append(result)
    return pd.DataFrame(output, columns=columns)


# The pool is scored once by the student; min and diversity pay for that pass.
SCORING_SELECTORS = {"min", "diversity"}
STUDENT_ZERO_SHOT = "gliner2.5-multi-v1"


def pool_size(dataset: str, locale: str, runs_root="runs") -> int:
    from .aggregate import read_json

    return len(read_json(Path(runs_root) / "splits" / f"{dataset}-{locale}.json")["pool"])


def annotation_usd(row, runs_root="runs") -> float | None:
    """Human labels at the stated rate; local teachers at measured free-GPU time."""
    from active_gliner import timing

    table = prices()
    if row["labels"] == "ground_truth":
        return row["n"] * table["human"]["usd_per_sentence"]
    teacher = table["teachers"].get(row["labels"])
    if not teacher or teacher["provider"] != "local":
        return None
    seconds = timing.seconds_per_sentence(row["labels"], row["dataset"], row["locale"], runs_root)
    if seconds is None:
        return None
    return row["n"] * seconds / 3600 * table["gpu"]["usd_per_hour"]


def selection_usd(row, df, runs_root="runs") -> float | None:
    """Pool size times the student's measured zero-shot time per sentence."""
    if row["selector"] not in SCORING_SELECTORS:
        return 0.0
    if "zero_shot_name" not in df:
        return None
    match = df[
        (df.zero_shot_name == STUDENT_ZERO_SHOT)
        & (df.dataset == row["dataset"])
        & (df.locale == row["locale"])
    ]
    if match.empty or pd.isna(match.iloc[0].get("latency_ms")):
        return None
    seconds = pool_size(row["dataset"], row["locale"], runs_root) * match.iloc[0].latency_ms / 1000
    return seconds / 3600 * prices()["gpu"]["usd_per_hour"]


def labelling_cost(df, runs_root="runs") -> pd.Series:
    """Annotation plus selection cost in USD per run; NaN when a measurement is missing."""
    values = []
    for row in df.to_dict("records"):
        annotation = annotation_usd(row, runs_root)
        selection = selection_usd(row, df, runs_root)
        values.append(
            annotation + selection
            if annotation is not None and selection is not None
            else float("nan")
        )
    return pd.Series(values, index=df.index, dtype=float)
