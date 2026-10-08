"""How few human labels beat the LLM teacher: the whole-pool replacement curve.

Every run trains on the whole pool. H sentences carry human labels (a nested random
subset), the rest carry teacher labels. H = 0 is teacher labels only, H = pool is human
labels only. Design: docs/research/2026-10-02-1207-minimum-human-labels-design.md.

The count is chosen on dev: the smallest H whose seed-mean dev F1 beats the teacher's dev
F1 with a paired bootstrap lower bound above zero. Test is read once, at that H only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import sentence_counts, test_ids
from .figures import _save
from .stats import holm, micro_f1
from .tables import DATASETS, MAIN_TEACHER


def minimum_rows(df: pd.DataFrame, pools: dict) -> pd.DataFrame:
    """Whole-pool runs with a `human` column: the number of human-labelled sentences."""
    rows = df[(df.selector == "all") & (df.variant == "long")].copy()
    if rows.empty:
        return rows.assign(human=pd.Series(dtype=int))
    for column in ("gt_fraction", "gt_assignment"):
        if column not in rows:
            rows[column] = None
    fraction = pd.to_numeric(rows.gt_fraction, errors="coerce")
    teacher_only = (rows.labels == MAIN_TEACHER) & fraction.isna()
    nested = (rows.labels == MAIN_TEACHER) & (rows.gt_assignment == "nested")
    human_only = rows.labels == "ground_truth"
    rows = rows[teacher_only | nested | human_only].copy()
    fraction = pd.to_numeric(rows.gt_fraction, errors="coerce").fillna(0.0)
    pool = [pools[(d, loc)] for d, loc in zip(rows.dataset, rows.locale, strict=True)]
    rows["human"] = (fraction * np.asarray(pool)).round().astype(int)
    rows.loc[rows.labels == "ground_truth", "human"] = np.asarray(pool)[
        (rows.labels == "ground_truth").to_numpy()
    ]
    return rows


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "locale", "human"]
    if rows.empty:
        return pd.DataFrame(columns=[*keys, "mean", "sd", "dev", "seeds"])
    grouped = rows.groupby(keys)
    table = grouped.f1.agg(mean="mean", sd="std", seeds="count")
    table["dev"] = 100 * grouped.dev_f1.mean()
    return table.reset_index()


def paired_gap(student_counts: list, teacher_counts, n_boot=1000, seed=0) -> dict:
    """Seed-mean student F1 minus teacher F1 (points), resampling sentences together."""
    teacher_counts = np.asarray(teacher_counts)
    rng = np.random.default_rng(seed)

    def gap(indices):
        student = np.mean([micro_f1(c[indices]) for c in student_counts])
        return 100 * (student - micro_f1(teacher_counts[indices]))

    size = len(teacher_counts)
    samples = np.asarray([gap(rng.integers(size, size=size)) for _ in range(n_boot)])
    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(
        gap=gap(np.arange(size)),
        low=float(low),
        high=float(high),
        p_value=float(min(1.0, 2 * np.mean(samples <= 0))),
    )


def _student_counts(run_dirs, split):
    ids = test_ids(run_dirs[0], split)
    for run_dir in run_dirs[1:]:
        if test_ids(run_dir, split) != ids:
            raise ValueError(f"Seeds predict different {split} sentences: {run_dir}")
    return ids, [sentence_counts(run_dir, split) for run_dir in run_dirs]


def choose(rows: pd.DataFrame, teacher_counts, n_boot=1000) -> dict:
    """Pick the fewest human labels on dev for every dataset; confirm once on test.

    Rows carry `human` (human-labelled sentences) and, optionally, `total` (all training
    sentences). Without `total` every run uses the whole pool, and the human-only whole
    pool is left out. Candidates are tried by fewest human labels, then fewest sentences.
    `teacher_counts(dataset, locale, split, ids)` returns the teacher's per-sentence counts.
    A dataset without a dev crossing reports none and is never read on test.
    """
    out = {}
    for (dataset, locale), local in rows.groupby(["dataset", "locale"]):
        local = local.copy()
        if "total" not in local:
            pool = local.human.max()
            local = local[local.human < pool].assign(total=pool)
        candidates = sorted(set(zip(local.human, local.total, strict=True)))
        result = dict(human=None, share=None)
        for human, total in candidates:
            cell = local[(local.human == human) & (local.total == total)]
            run_dirs = sorted(cell.run_dir)
            try:
                ids, student = _student_counts(run_dirs, "dev")
                teacher = teacher_counts(dataset, locale, "dev", ids)
            except FileNotFoundError as exc:
                result["missing"] = str(exc)
                break
            dev = paired_gap(student, teacher, n_boot)
            if dev["gap"] > 0 and dev["low"] > 0:
                ids, student = _student_counts(run_dirs, "test")
                test = paired_gap(student, teacher_counts(dataset, locale, "test", ids), n_boot)
                result = dict(
                    human=int(human),
                    total=int(total),
                    share=human / total,
                    seeds=len(run_dirs),
                    dev=dev,
                    test=test,
                )
                if "f1" in cell:
                    result["f1"] = float(cell.f1.mean())
                break
        out[(dataset, locale)] = result
    confirmed = [key for key, result in out.items() if "test" in result]
    if confirmed:
        adjusted = holm([out[key]["test"]["p_value"] for key in confirmed])
        for key, p in zip(confirmed, adjusted, strict=True):
            out[key]["test"]["p_holm"] = p
    return out


HUMAN_NAMES = {
    0: "Zero",
    25: "TwentyFive",
    50: "Fifty",
    100: "Hundred",
    200: "TwoHundred",
    400: "FourHundred",
    1000: "Thousand",
    2500: "TwentyFiveHundred",
}


def macros(table: pd.DataFrame, chosen: dict) -> dict:
    """Test F1 per human count (WholePool = human labels only) and the dev-chosen count."""
    out = {}
    for (dataset, locale), local in table.groupby(["dataset", "locale"]):
        if (dataset, locale) not in DATASETS:
            continue
        pool = local.human.max()
        for row in local.to_dict("records"):
            human = "WholePool" if row["human"] == pool else HUMAN_NAMES.get(row["human"])
            if human:
                out[DATASETS[(dataset, locale)][1] + "MinimumAt" + human] = row["mean"]
    for key, result in chosen.items():
        if key not in DATASETS or "test" not in result:
            continue
        name = DATASETS[key][1] + "Minimum"
        out[name + "Human"] = result["human"]
        out[name + "Share"] = 100 * result["share"]
        out[name + "TestGap"] = result["test"]["gap"]
        out[name + "TestLow"] = result["test"]["low"]
        out[name + "TestHigh"] = result["test"]["high"]
    return out


def figure(
    table: pd.DataFrame, teacher_f1: dict, chosen: dict, out, human_only=None
) -> Path | None:
    """Test F1 against human-labelled sentences; every run uses the whole pool.

    `human_only` (dataset, locale, budget, mean) adds the same human counts without the
    teacher-labelled rest: random human sentences alone, from the mixing curves.
    """
    places = [key for key in DATASETS if key in set(zip(table.dataset, table.locale, strict=True))]
    if not places:
        return None
    columns = 4 if len(places) > 3 else len(places)
    lines = (len(places) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(3.4 * columns, 2.9 * lines), squeeze=False)
    for ax, key in zip(axes.flat, places, strict=False):
        local = table[(table.dataset == key[0]) & (table.locale == key[1])].sort_values("human")
        # H = 0 cannot sit on a log axis; draw it at 10 and label it.
        x = local.human.clip(lower=10)
        ax.errorbar(
            x,
            local["mean"],
            yerr=local.sd.fillna(0),
            marker="o",
            markersize=3,
            label="human + teacher labels on the rest",
        )
        if human_only is not None:
            alone = human_only[
                (human_only.dataset == key[0]) & (human_only.locale == key[1])
            ].copy()
            alone["x"] = pd.to_numeric(alone.budget, errors="coerce")
            alone = alone.dropna(subset=["x"])
            # The whole pool with human labels ends both lines.
            whole = local[local.human == local.human.max()].assign(x=lambda t: t.human)
            alone = pd.concat([alone, whole]).sort_values("x")
            ax.plot(
                alone.x,
                alone["mean"],
                marker="s",
                markersize=3,
                color="tab:green",
                label="human labels only",
            )
        teacher = teacher_f1.get(key)
        if teacher is not None:
            ax.axhline(teacher, color="red", linestyle="--", linewidth=1, label="teacher LLM")
        human = chosen.get(key, {}).get("human")
        if human:
            ax.axvline(human, color="gray", linestyle=":", linewidth=1, label="chosen on dev")
        ax.set_xscale("log")
        ticks = [10, 100, 1000, 10000]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "100", "1000", "10000"])
        ax.set_title(DATASETS[key][0], fontsize="medium")
        ax.set_xlabel("Human-labelled sentences (log)", fontsize="small")
        ax.set_ylabel("Test micro-F1 (%)", fontsize="small")
        ax.tick_params(labelsize="x-small")
    for ax in list(axes.flat)[len(places) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize="x-small")
    return _save(fig, "minimum-human-labels", out)
