"""Paper plots with explicit units and observed, un-interpolated points."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator

from . import style  # noqa: F401  (sets the figure style)


def _save(fig, name, out):
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    # No creation date in the PDF, so a rerun on the same runs gives the same file.
    fig.savefig(path / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(path / f"{name}.png", dpi=180)
    plt.close(fig)
    return path / f"{name}.pdf"


def pareto_front(points, x, y) -> list:
    points = pd.DataFrame(points).dropna(subset=[x, y]).sort_values(x, kind="stable")
    return [
        row["name"]
        for _, row in points.iterrows()
        if not (
            (points[x] <= row[x])
            & (points[y] >= row[y])
            & ((points[x] < row[x]) | (points[y] > row[y]))
        ).any()
    ]


def _curve_rows(df, dataset):
    """The main arms only: no mixing, the zero no-prediction rule, no whole-pool run."""
    rows = df[(df.dataset == dataset) & df.labels.isin(["ground_truth", "gemma-4-12b"])]
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    if "no_prediction" in rows:
        rows = rows[rows.no_prediction.fillna("zero") == "zero"]
    rows = rows[rows.selector != "all"]
    if "locale" not in rows:
        rows = rows.assign(locale="en-US")
    return rows


def cost_curves(df, dataset, out="paper/figures") -> list[Path]:
    """Test F1 against labelling cost; needs a `labelling_cost_usd` column."""
    rows = _curve_rows(df, dataset).dropna(subset=["labelling_cost_usd"])
    rows = rows[rows.labelling_cost_usd > 0]
    paths = []
    for locale, group in rows.groupby("locale"):
        fig, ax = plt.subplots()
        colors = dict(
            zip(sorted(group.selector.unique()), plt.get_cmap("tab10").colors, strict=False)
        )
        for (label, selector), curve in group.groupby(["labels", "selector"]):
            summary = curve.groupby("n")[["labelling_cost_usd", "f1"]].agg(["mean", "std"])
            ax.errorbar(
                summary["labelling_cost_usd"]["mean"],
                summary["f1"]["mean"],
                yerr=summary["f1"]["std"].fillna(0),
                linestyle="-" if label == "ground_truth" else "--",
                marker="o",
                color=colors[selector],
                label=f"{selector}, {label}",
            )
        ax.set_xscale("log")
        ax.set(
            xlabel="Labelling cost (USD, estimated; log scale)",
            ylabel="Test micro-F1 (%)",
            ylim=(0, 100),
            title=f"{dataset}, {locale} (error bars: seed SD)",
        )
        ax.legend(fontsize="small")
        paths.append(_save(fig, f"cost-{dataset}-{locale}", out))
    return paths


def _reference_lines(ax, df, dataset, locale) -> None:
    """Context only (design): the whole-pool runs above, the zero-shot student below."""
    local = df[(df.dataset == dataset) & (df.get("locale", "en-US") == locale)]
    for label, dash in (("ground_truth", "-"), ("gemma-4-12b", "--")):
        whole = local[(local.selector == "all") & (local.labels == label)]
        if not whole.empty:
            ax.axhline(
                whole.f1.mean(),
                color="gray",
                linestyle=dash,
                linewidth=1,
                label=f"whole pool, {label}",
            )
    if "zero_shot_name" in local:
        zero = local[local.zero_shot_name == "gliner2.5-multi-v1"]
        if not zero.empty:
            ax.axhline(
                zero.f1.mean(),
                color="black",
                linestyle=":",
                linewidth=1,
                label="zero-shot student",
            )


def learning_curves(df, dataset, out="paper/figures") -> list[Path]:
    rows = _curve_rows(df, dataset)
    paths = []
    for locale, group in rows.groupby("locale"):
        fig, ax = plt.subplots()
        colors = dict(
            zip(sorted(group.selector.unique()), plt.get_cmap("tab10").colors, strict=False)
        )
        for (label, selector), curve in group.groupby(["labels", "selector"]):
            summary = curve.groupby("n").f1.agg(["mean", "std"]).sort_index()
            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                linestyle="-" if label == "ground_truth" else "--",
                marker="o",
                color=colors[selector],
                label=f"{selector}, {label}",
            )
        _reference_lines(ax, df, dataset, locale)
        ax.set(
            xlabel="Labelled sentences (N)",
            ylabel="Test micro-F1 (%)",
            ylim=(0, 100),
            title=f"{dataset}, {locale} (error bars: seed SD)",
        )
        ax.legend(fontsize="small")
        paths.append(_save(fig, f"learning-{dataset}-{locale}", out))
    return paths


TASK_NAMES = {
    "ner": "NER",
    "relations": "Relations",
    "classification": "Multi-label",
    "slots": "Slots",
}


def interaction_forest(report, out="paper/figures") -> Path | None:
    """One row per task: the interaction and its 95% interval, coloured by task."""
    rows = [(task, entry) for task, entry in report["interactions"].items() if "mean" in entry]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH, 0.42 * len(rows) + 0.55))
    ax.axvline(0, color=style.FRAME, linewidth=0.9, zorder=1)
    for i, (task, entry) in enumerate(rows):
        name = TASK_NAMES.get(task, task)
        color = style.TASK_COLORS.get(name, style.BLUE)
        interval = entry.get("interval", {})
        if "low" in interval:
            ax.hlines(i, interval["low"], interval["high"], color=color, linewidth=2.6, alpha=0.35)
        ax.plot(entry["mean"], i, "o", color=color, markersize=6.5, zorder=3)
        ax.annotate(
            f"{entry['mean']:+.1f}",
            (entry["mean"], i),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=style.INK,
        )
    ax.set_yticks(range(len(rows)), [TASK_NAMES.get(task, task) for task, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Interaction (F1 points)")
    return _save(fig, "interaction-forest", out)


def _points(points, x):
    points = pd.DataFrame(points).copy()
    points = points.dropna(subset=[x, "f1"])
    if (points[x] <= 0).any():
        raise ValueError("Log axes require strictly positive cost or latency")
    for key in ("coverage", "timing_protocol"):
        if key in points and points[key].nunique(dropna=False) > 1:
            raise ValueError(f"Separate panels are required for different {key}")
    return points


def _bubbles(ax, points, x):
    if "family" not in points:
        points = points.assign(family="unspecified")
    for index, (family, group) in enumerate(points.groupby("family", dropna=False)):
        color = plt.get_cmap("tab10")(index % 10)
        sizes = pd.to_numeric(group.get("params", pd.Series(np.nan, index=group.index)))
        known = sizes.notna() & (sizes > 0)
        ax.scatter(
            group.loc[known, x],
            group.loc[known, "f1"],
            s=sizes[known] * 20,
            label=family,
            alpha=0.7,
            color=color,
        )
        if (~known).any():
            ax.scatter(
                group.loc[~known, x],
                group.loc[~known, "f1"],
                marker="x",
                s=60,
                label=f"{family}, unknown size",
                color=color,
            )
    for _, row in points.iterrows():
        # Zero-shot models sit close together; labels below them keep clear of the students.
        below = str(row.get("family", "")).startswith("zero-shot")
        ax.annotate(
            row["name"],
            (row[x], row.f1),
            xytext=(5, -12 if below else 5),
            textcoords="offset points",
            fontsize=7,
        )
        if pd.notna(row.get("sd")):
            ax.errorbar(row[x], row.f1, yerr=row["sd"], color="gray", capsize=3)
    ax.set_xscale("log")
    # Start the axis one step of 10 below the lowest point, so the points spread out.
    ax.set_ylim(max(0, 10 * np.floor(points.f1.min() / 10) - 10), 100)
    ax.set_ylabel("Test micro-F1 (%)")
    ax.legend(fontsize="x-small")


def pareto_chart(points, out="paper/figures") -> Path | None:
    """cost is inference-only estimated USD per 1M sentences; params is billions."""
    points = _points(points, "cost")
    if points.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    _bubbles(ax, points, "cost")
    names = pareto_front(points, "cost", "f1")
    front = points[points.name.isin(names)].sort_values("cost")
    ax.plot(front.cost, front.f1, linestyle="--", color="gray")
    ax.set(
        xlabel="Inference cost per 1M sentences (USD, estimated; log scale)",
    )
    return _save(fig, "pareto", out)


ZERO_SHOT_NAMES = {
    "gliner2.5-small-v1": "GLiNER2.5 small",
    "gliner2.5-base-v1": "GLiNER2.5 base",
    "gliner2.5-multi-v1": "GLiNER2.5 multi (our student)",
    "gliner2-large-v1": "GLiNER2 large",
}
STUDENT_NAMES = {
    ("ground_truth", "random"): "human labels, random",
    ("gemma-4-12b", "random"): "Gemma labels, random",
    ("gemma-4-12b", "min"): "Gemma labels, min",
}


def quadrant_chart(points, out="paper/figures") -> Path | None:
    """Speed against quality for NER: zero-shot models, trained students, the LLM teacher."""
    points = _points(points, "latency_ms")
    if points.empty:
        return None
    family = points.get("family", pd.Series("", index=points.index)).fillna("")
    zero = points[family.str.startswith("zero-shot")]
    students = points[family == "fine-tuned student"]
    teachers = points[family.str.contains("teacher")]
    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH, 2.6))
    ax.set_xscale("log")
    ax.set_xlim(1.2, 1500)
    ax.set_ylim(50, 92)
    ax.xaxis.set_major_formatter(FuncFormatter(style.compact))
    ax.xaxis.set_minor_locator(NullLocator())
    base = zero[zero.name == "gliner2.5-multi-v1"]
    best = students.sort_values("f1").iloc[-1] if not students.empty else None
    if not base.empty and best is not None:
        ax.annotate(
            "",
            xy=(best.latency_ms, best.f1),
            xytext=(base.iloc[0].latency_ms, base.iloc[0].f1),
            arrowprops={
                "arrowstyle": "-|>",
                "color": style.LEADER,
                "linewidth": 0.8,
                "shrinkA": 4,
                "shrinkB": 5,
            },
        )
    ax.scatter(
        zero.latency_ms,
        zero.f1,
        s=28,
        color="#b7bfcd",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="zero-shot GLiNER",
    )
    ax.scatter(
        students.latency_ms,
        students.f1,
        s=34,
        color=style.BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="fine-tuned student",
    )
    if "sd" in students:
        for _, row in students.dropna(subset=["sd"]).iterrows():
            ax.vlines(
                row.latency_ms,
                row.f1 - row.sd,
                row.f1 + row.sd,
                color=style.BLUE,
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
    ax.scatter(
        teachers.latency_ms,
        teachers.f1,
        s=90,
        marker="*",
        color=style.TEACHER,
        edgecolor=style.INK,
        linewidth=0.6,
        zorder=4,
        label="LLM teacher",
    )
    # Names: zero-shot models and students sit near 2 ms, so their labels go to the right
    # in two columns, spread apart so they never overlap.
    zero_y = style.spread(list(zero.f1), gap=2.6)
    for (_, row), y in zip(zero.iterrows(), zero_y, strict=True):
        name = ZERO_SHOT_NAMES.get(row["name"], row["name"])
        style.label_at(ax, (row.latency_ms, row.f1), (9, y), name)
    student_y = style.spread(list(students.f1), gap=3.2)
    for (_, row), y in zip(students.iterrows(), student_y, strict=True):
        name = STUDENT_NAMES.get((row.get("labels"), row.get("selector")), row["name"])
        style.label_at(ax, (row.latency_ms, row.f1), (4, y), name)
    for _, row in teachers.iterrows():
        style.label_at(
            ax, (row.latency_ms, row.f1), (row.latency_ms, row.f1 - 6), "Gemma 4 12B", ha="center"
        )
    ax.set_xlabel("Latency (ms per sentence, log scale)")
    ax.set_ylabel("Test micro-F1 (%)")
    return _save(fig, "quadrants", out)


NAME_TASKS = {
    "CleanCoNLL": "NER",
    "BC5CDR": "NER",
    "MIT Movie": "NER",
    "CrossRE": "Relations",
    "Hallmarks": "Multi-label",
    "MASSIVE en": "Slots",
    "MASSIVE fr": "Slots",
}


def teacher_bins_chart(tables, out="paper/figures") -> Path | None:
    """Teacher F1 by the student's confidence group; lines coloured by task, named at the end."""
    if not tables:
        return None
    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH, 2.4))
    ends = []
    for name, table in tables.items():
        table = pd.DataFrame(table).sort_values("bin")
        color = style.TASK_COLORS.get(NAME_TASKS.get(name, ""), style.BLUE)
        ax.plot(table["bin"], 100 * table.f1, marker="o", color=color, linewidth=1.6)
        ends.append((name, table["bin"].iloc[-1], 100 * table.f1.iloc[-1]))
    places = style.spread([y for _, _, y in ends], gap=5.0)
    for (name, x, y), place in zip(ends, places, strict=True):
        style.label_at(ax, (x, y), (x + 0.35, place), name)
    bins = sorted(pd.DataFrame(next(iter(tables.values())))["bin"])
    names = [str(b + 1) for b in bins]
    names[0] += "\nleast sure"
    names[-1] += "\nmost sure"
    ax.set_xticks(bins, names)
    # Room on the right for the dataset names, inside the axes so the figure keeps its size.
    ax.set_xlim(bins[0] - 0.25, bins[-1] + 1.45)
    ax.set(
        xlabel="Untrained student's confidence (four equal groups)",
        ylabel="Teacher micro-F1 (%)",
        ylim=(0, 100),
    )
    return _save(fig, "teacher-bins", out)
