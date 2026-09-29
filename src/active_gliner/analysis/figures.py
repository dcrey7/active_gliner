"""Paper plots with explicit units and observed, un-interpolated points."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, name, out):
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # No creation date in the PDF, so a rerun on the same runs gives the same file.
    fig.savefig(path / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path / f"{name}.png", dpi=180, bbox_inches="tight")
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
    for label, style in (("ground_truth", "-"), ("gemma-4-12b", "--")):
        whole = local[(local.selector == "all") & (local.labels == label)]
        if not whole.empty:
            ax.axhline(
                whole.f1.mean(),
                color="gray",
                linestyle=style,
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


def interaction_forest(report, out="paper/figures") -> Path | None:
    rows = [(task, entry) for task, entry in report["interactions"].items() if "mean" in entry]
    if not rows:
        return None
    fig, ax = plt.subplots()
    for i, (_, entry) in enumerate(rows):
        ax.plot(entry["mean"], i, "o", color="tab:blue")
        interval = entry.get("interval", {})
        if "low" in interval:
            ax.hlines(i, interval["low"], interval["high"], color="tab:blue")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(
        yticks=range(len(rows)),
        yticklabels=[name for name, _ in rows],
        xlabel="Interaction (F1 points)",
        title="95% paired intervals: test uncertainty only",
    )
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
            label=f"{family} (area: parameters in billions)",
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
            fontsize=8,
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
    fig, ax = plt.subplots()
    _bubbles(ax, points, "cost")
    names = pareto_front(points, "cost", "f1")
    front = points[points.name.isin(names)].sort_values("cost")
    ax.plot(front.cost, front.f1, linestyle="--", color="gray")
    ax.set(
        xlabel="Inference cost per 1M sentences (USD, estimated; log scale)",
        title="Observed point-estimate front",
    )
    return _save(fig, "pareto", out)


def quadrant_chart(points, out="paper/figures") -> Path | None:
    points = _points(points, "latency_ms")
    if points.empty:
        return None
    fig, ax = plt.subplots()
    _bubbles(ax, points, "latency_ms")
    ax.axvline(points.latency_ms.median(), linestyle=":", color="gray")
    ax.axhline(points.f1.median(), linestyle=":", color="gray")
    for _, row in points.iterrows():
        source = points[points.name == row.get("zero_shot_name")]
        if len(source) == 1:
            start = source.iloc[0]
            # The end point's own name states the selector and label source.
            ax.annotate(
                "",
                xy=(row.latency_ms, row.f1),
                xytext=(start.latency_ms, start.f1),
                arrowprops={"arrowstyle": "->", "color": "black"},
            )
    ax.set(xlabel="Latency (ms per sentence, log scale)", title="Descriptive median splits")
    return _save(fig, "quadrants", out)


def teacher_bins_chart(tables, out="paper/figures") -> Path | None:
    if not tables:
        return None
    fig, ax = plt.subplots()
    for name, table in tables.items():
        table = pd.DataFrame(table)
        ax.plot(table["bin"], 100 * table.f1, marker="o", label=name.replace("/", ", "))
    bins = sorted(pd.DataFrame(next(iter(tables.values())))["bin"])
    names = [str(b + 1) for b in bins]
    names[0] += "\nleast sure"
    names[-1] += "\nmost sure"
    ax.set_xticks(bins, names)
    ax.set(
        xlabel="Student confidence group (equal-size groups; ties split by stable rank)",
        ylabel="Teacher micro-F1 (%)",
        ylim=(0, 100),
    )
    # Outside the axes, so no line is hidden.
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    return _save(fig, "teacher-bins", out)


def calibration_chart(run_dirs, out="paper/figures") -> Path | None:
    available = [Path(path) for path in run_dirs if (Path(path) / "calibration.csv").exists()]
    if not available:
        return None
    fig, ax = plt.subplots()
    for directory in available:
        rows = pd.read_csv(directory / "calibration.csv")
        rows = rows[rows["count"] > 0]
        ax.plot(rows.mean_confidence, rows.correctness, marker="o", label=str(directory))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="Mean confidence",
        ylabel="Empirical correctness",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Descriptive calibration; extraction bins contain emitted items only",
    )
    ax.legend(fontsize="xx-small")
    return _save(fig, "calibration", out)
