"""Does the student's confidence tell when it is right? Before and after training.

The thesis showed this for MIT Movie with four wide bins. Here: ten bins, every dataset,
the zero-shot student (the model that selects sentences) and the trained student (400
random ground-truth sentences, seeds pooled). Test split, descriptive only.

Items follow evaluate.errors.items: emitted spans for extraction (correct = the span is
in the ground truth), every label decision for classification and relation pairs
(correct = the label is in the ground truth).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from active_gliner import data, tasks
from active_gliner.evaluate.errors import items

from .figures import _save
from .tables import DATASETS, DEFAULT_BUDGET, MAIN_BUDGET, ZERO_SHOT_STUDENT

BINS = 10


def item_pairs(task, records, predictions) -> list[tuple[float, bool]]:
    """(confidence, correct) for every scored item."""
    pairs = []
    for record, prediction in zip(records, predictions, strict=True):
        gold = task.gold_items(record)
        for key, _item, confidence in items(task, prediction):
            pairs.append((float(confidence), key in gold))
    return pairs


def binned(pairs) -> pd.DataFrame:
    """Count, mean confidence and share correct in ten equal-width confidence bins."""
    frame = pd.DataFrame(pairs, columns=["confidence", "correct"])
    frame["bin"] = (frame.confidence * BINS).astype(int).clip(upper=BINS - 1)
    grouped = frame.groupby("bin")
    table = pd.DataFrame(
        {
            "count": grouped.size(),
            "mean_confidence": grouped.confidence.mean(),
            "correct": grouped.correct.mean(),
        }
    ).reindex(range(BINS))
    table["count"] = table["count"].fillna(0).astype(int)
    table.index.name = "bin"
    return table.reset_index()


def run_pairs(run_dir, dataset: str, locale: str, split="test") -> list[tuple[float, bool]]:
    path = Path(run_dir) / "predictions" / f"{split}.jsonl"
    if not path.exists():
        return []
    by_id = {r.id: r for r in data.load(dataset, locale)[split]}
    predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
    records = [by_id[p["id"]] for p in predictions]
    return item_pairs(tasks.for_dataset(dataset), records, predictions)


def trained_dirs(df: pd.DataFrame, dataset: str, locale: str) -> list[str]:
    """Ground truth, random, main budget: the trained student of the main table."""
    rows = df[
        (df.dataset == dataset)
        & (df.locale == locale)
        & (df.labels == "ground_truth")
        & (df.selector == "random")
        & (df.n == MAIN_BUDGET.get(dataset, DEFAULT_BUDGET))
    ]
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    if "variant" in rows:
        rows = rows[rows.variant.isna()]
    return list(rows.run_dir)


def tables(df: pd.DataFrame, runs="runs") -> dict:
    """{(dataset, locale): {"zero-shot": table, "trained": table}} for every dataset."""
    out = {}
    for dataset, locale in DATASETS:
        zero_dir = Path(runs) / "zero_shot" / ZERO_SHOT_STUDENT / dataset / locale
        models = {"zero-shot": run_pairs(zero_dir, dataset, locale)}
        trained = []
        for run_dir in trained_dirs(df, dataset, locale):
            trained.extend(run_pairs(run_dir, dataset, locale))
        models["trained"] = trained
        out[(dataset, locale)] = {name: binned(p) for name, p in models.items() if p}
    return out


def share_correct(table: pd.DataFrame, low: float, high: float) -> float | None:
    """Share correct over all items with confidence in [low, high)."""
    rows = table[(table.bin >= round(low * BINS)) & (table.bin < round(high * BINS))]
    rows = rows[rows["count"] > 0]
    if rows.empty:
        return None
    return float((rows.correct * rows["count"]).sum() / rows["count"].sum())


MIN_ITEMS = 20


def macros(all_tables: dict) -> dict:
    """Share correct (%) in the lowest bin with enough items and in the top bin.

    The lowest bin differs by dataset: extraction students emit nothing under their
    threshold, so Low comes with LowFrom, the lower edge of that bin.
    """
    out = {}
    names = {"zero-shot": "ZeroShot", "trained": "Trained"}
    for key, models in all_tables.items():
        for model, table in models.items():
            prefix = DATASETS[key][1] + "Reliability" + names[model]
            filled = table[(table["count"] >= MIN_ITEMS) & (table.bin < BINS - 1)]
            if not filled.empty:
                low = filled.iloc[0]
                out[prefix + "Low"] = 100 * float(low.correct)
                out[prefix + "LowFrom"] = int(low.bin) / BINS
            top = table[table.bin == BINS - 1].iloc[0]
            if top["count"] >= MIN_ITEMS:
                out[prefix + "High"] = 100 * float(top.correct)
    return out


def figure(all_tables: dict, out) -> Path | None:
    places = [key for key, models in all_tables.items() if models]
    if not places:
        return None
    columns = 4 if len(places) > 3 else len(places)
    lines = (len(places) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(3.2 * columns, 3.0 * lines), squeeze=False)
    styles = {"zero-shot": ("tab:orange", "o"), "trained": ("tab:blue", "s")}
    for ax, key in zip(axes.flat, places, strict=False):
        for model, table in all_tables[key].items():
            shown = table[table["count"] >= 20]
            color, marker = styles[model]
            ax.plot(
                shown.mean_confidence,
                shown.correct,
                color=color,
                marker=marker,
                markersize=3,
                label=model,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=DATASETS[key][0])
        ax.set_xlabel("Confidence", fontsize="small")
        ax.set_ylabel("Share correct", fontsize="small")
        ax.tick_params(labelsize="x-small")
    for ax in list(axes.flat)[len(places) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize="small")
    return _save(fig, "reliability", out)


def latex_table(all_tables: dict, path) -> Path:
    """Zero-shot share correct (%) per confidence bin from 0.5 up; counts under 20 shown as --."""
    header = " & ".join(f"{b / BINS:.1f}" for b in range(5, BINS))
    lines = [
        "\\begin{tabular}{l" + "r" * (BINS - 5) + "}",
        "\\toprule",
        "Dataset & " + header + " \\\\",
        "\\midrule",
    ]
    for key, models in all_tables.items():
        table = models.get("zero-shot")
        if table is None:
            continue
        cells = []
        for b in range(5, BINS):
            row = table[table.bin == b].iloc[0]
            cells.append("--" if row["count"] < 20 else f"{100 * row.correct:.0f}")
        lines.append(DATASETS[key][0] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
