"""Points for the deployment figures (Pareto and quadrant): NER, pooled over three datasets.

Every point uses free-GPU timings from 28 Sep 2026 on the same 3090:
GLiNER models at batch 8 (zero-shot runs), LLM teachers as whole-job wall time with
8 parallel requests (time-teacher). Fine-tuned students reuse the zero-shot timing of
their base model: LoRA adapters were not timed separately.
"""

import pandas as pd

from active_gliner import timing

from . import cost

NER_DATASETS = ("bc5cdr", "cleanconll", "mit_movie")
STUDENT_BASE = "gliner2.5-multi-v1"
TIMING_PROTOCOL = "RTX 3090, free GPU; GLiNER batch 8; LLM server 8 parallel requests"
# (labels, selector, N) arms shown as fine-tuned students.
STUDENT_ARMS = (
    ("ground_truth", "random", 400),
    ("gemma-4-12b", "random", 400),
    ("gemma-4-12b", "min", 400),
)
SHORT_LABELS = {"ground_truth": "true labels", "gemma-4-12b": "Gemma"}
# Nominal size in billions of parameters; None draws an "unknown size" marker.
TEACHER_PARAMS = {"gemma-4-12b": 12.0}


def usd_per_million(latency_ms: float) -> float:
    return latency_ms / 1000 * 1e6 / 3600 * cost.prices()["gpu"]["usd_per_hour"]


def _zero_shot(df) -> pd.DataFrame:
    if "zero_shot_name" not in df:
        return pd.DataFrame()
    rows = df[df.zero_shot_name.notna() & df.dataset.isin(NER_DATASETS)]
    return rows[rows.locale == "en-US"]


def zero_shot_points(df) -> list[dict]:
    points = []
    for name, group in _zero_shot(df).groupby("zero_shot_name"):
        if sorted(group.dataset) != sorted(NER_DATASETS):
            continue
        latency = group.latency_ms.mean()
        points.append(
            dict(
                name=name,
                family="zero-shot GLiNER",
                params=group.params.iloc[0] / 1e9,
                f1=group.f1.mean(),
                latency_ms=latency,
                cost=usd_per_million(latency),
            )
        )
    return points


def student_points(df) -> list[dict]:
    base = _zero_shot(df)
    base = base[base.zero_shot_name == STUDENT_BASE]
    if sorted(base.dataset) != sorted(NER_DATASETS):
        return []
    latency = base.latency_ms.mean()
    points = []
    for labels, selector, n in STUDENT_ARMS:
        rows = df[
            df.dataset.isin(NER_DATASETS)
            & (df.labels == labels)
            & (df.selector == selector)
            & (df.n == n)
            & (df.locale == "en-US")
        ]
        if "gt_fraction" in rows:
            rows = rows[rows.gt_fraction.isna()]
        if "no_prediction" in rows:
            rows = rows[rows.no_prediction.fillna("zero") == "zero"]
        if rows.dataset.nunique() != len(NER_DATASETS):
            continue
        # Equal weight per dataset, over seeds that every dataset has.
        seeds = set.intersection(*(set(g.seed) for _, g in rows.groupby("dataset")))
        rows = rows[rows.seed.isin(seeds)]
        if rows.dataset.nunique() != len(NER_DATASETS):
            continue
        per_seed = rows.groupby("seed").f1.mean()
        points.append(
            dict(
                name=f"student: {SHORT_LABELS[labels]}, {selector}",
                family="fine-tuned student",
                params=base.params.iloc[0] / 1e9,
                f1=per_seed.mean(),
                sd=per_seed.std() if len(per_seed) > 1 else None,
                latency_ms=latency,
                cost=usd_per_million(latency),
                zero_shot_name=STUDENT_BASE,
                labels=labels,
                selector=selector,
                n=n,
            )
        )
    return points


def teacher_points(teacher_scores: dict, runs_root="runs") -> list[dict]:
    points = []
    for teacher, params in TEACHER_PARAMS.items():
        f1s, seconds = [], []
        for dataset in NER_DATASETS:
            score = teacher_scores.get(f"{dataset}/en-US/{teacher}")
            spent = timing.seconds_per_sentence(teacher, dataset, "en-US", runs_root)
            if score is None or spent is None:
                break
            f1s.append(score["f1"])
            seconds.append(spent)
        else:
            latency = 1000 * sum(seconds) / len(seconds)
            points.append(
                dict(
                    name=teacher,
                    family="local LLM teacher",
                    params=params,
                    f1=sum(f1s) / len(f1s),
                    latency_ms=latency,
                    cost=usd_per_million(latency),
                )
            )
    return points


def macros(points: pd.DataFrame) -> dict:
    """Paper macros for the deployment points: F1, latency and cost per 1M sentences."""
    prefixes = {STUDENT_BASE: "DeployZeroShot"}
    prefixes.update({name: "DeployTeacher" for name in TEACHER_PARAMS})
    for labels, selector, _ in STUDENT_ARMS:
        source = "True" if labels == "ground_truth" else SHORT_LABELS[labels]
        name = f"student: {SHORT_LABELS[labels]}, {selector}"
        prefixes[name] = "DeployStudent" + source + selector.capitalize()
    values = {}
    for row in points.to_dict("records") if not points.empty else []:
        prefix = prefixes.get(row["name"])
        if prefix is None:
            continue
        values[prefix] = row["f1"]
        values[prefix + "Latency"] = row["latency_ms"]
        values[prefix + "Cost"] = row["cost"]
    if "DeployTeacherLatency" in values and "DeployStudentGemmaRandomLatency" in values:
        speedup = values["DeployTeacherLatency"] / values["DeployStudentGemmaRandomLatency"]
        values["DeploySpeedup"] = int(round(speedup))
    return values


def points(df, teacher_scores: dict, runs_root="runs") -> pd.DataFrame:
    rows = zero_shot_points(df) + student_points(df) + teacher_points(teacher_scores, runs_root)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table["task"] = "ner"
    table["locale"] = "en-US"
    table["coverage"] = ",".join(NER_DATASETS)
    table["timing_protocol"] = TIMING_PROTOCOL
    return table
