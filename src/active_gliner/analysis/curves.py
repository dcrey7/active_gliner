"""The mixing curves (thesis Fig 4.6.1 on every dataset): test F1 against labelled sentences.

One line per human share (0% = teacher labels only, 100% = human labels only), the
teacher's own test F1 as the line to beat, and random selection as the comparison.
Design: docs/research/2026-10-02-1207-minimum-human-labels-design.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .figures import _save
from .tables import DATASETS, MAIN_TEACHER

SHARES = {0: "Zero", 25: "TwentyFive", 50: "Fifty", 75: "SeventyFive", 100: "Hundred"}
BUDGETS = {100: "Hundred", 400: "FourHundred", 1000: "Thousand", 2500: "TwentyFiveHundred"}
WHOLE = "WholePool"


def curve_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Curve cells from every block: ranked (min, empty last) and random, plus whole pool.

    Columns added: rule ("ranked" or "random"), share (% human labels), budget (N, or
    "all" for the whole pool).
    """
    rows = df[df.labels.isin(["ground_truth", MAIN_TEACHER])].copy()
    for column, default in (
        ("gt_fraction", None),
        ("gt_assignment", None),
        ("no_prediction", "zero"),
        ("variant", None),
    ):
        if column not in rows:
            rows[column] = default
    rows["no_prediction"] = rows.no_prediction.fillna("zero")
    mixed_ok = rows.gt_fraction.isna() | (rows.gt_assignment == "random")
    ranked = (rows.selector == "min") & (rows.no_prediction == "last") & rows.variant.isna()
    random = (rows.selector == "random") & rows.gt_fraction.isna() & rows.variant.isna()
    whole = (rows.selector == "all") & (rows.variant == "long")
    rows = rows[mixed_ok & (ranked | random | whole)].copy()
    rows["rule"] = "ranked"
    rows.loc[rows.selector == "random", "rule"] = "random"
    fraction = pd.to_numeric(rows.gt_fraction, errors="coerce").fillna(0.0)
    rows["share"] = (100 * fraction).round().astype(int)
    rows.loc[rows.labels == "ground_truth", "share"] = 100
    rows["budget"] = rows.n.astype(object)
    rows.loc[rows.selector == "all", "budget"] = "all"
    return rows


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and seed count per dataset, locale, rule, share and budget."""
    keys = ["dataset", "locale", "rule", "share", "budget"]
    if rows.empty:
        return pd.DataFrame(columns=[*keys, "mean", "sd", "seeds"])
    grouped = rows.groupby(keys, dropna=False).f1
    return grouped.agg(mean="mean", sd="std", seeds="count").reset_index()


def macros(table: pd.DataFrame) -> dict:
    """Ranked cells as macros: dataset + Curve + share + At + budget (mean test F1)."""
    out = {}
    for row in table[table.rule == "ranked"].to_dict("records"):
        key = (row["dataset"], row["locale"])
        budget = WHOLE if row["budget"] == "all" else BUDGETS.get(row["budget"])
        if key not in DATASETS or budget is None or row["share"] not in SHARES:
            continue
        name = DATASETS[key][1] + "Curve" + SHARES[row["share"]] + "At" + budget
        out[name] = row["mean"]
    return out


def _x(budget, pool: int) -> float:
    return pool if budget == "all" else float(budget)


def figure(rows: pd.DataFrame, teacher_f1: dict, pools: dict, out) -> Path | None:
    """One panel per dataset; x is labelled sentences on a log scale (whole pool last)."""
    table = summary(rows)
    places = [key for key in DATASETS if key in set(zip(table.dataset, table.locale, strict=True))]
    if not places:
        return None
    columns = 4 if len(places) > 3 else len(places)
    lines = (len(places) + columns - 1) // columns
    fig, axes = plt.subplots(lines, columns, figsize=(3.4 * columns, 2.9 * lines), squeeze=False)
    colors = plt.get_cmap("viridis")
    for ax, (dataset, locale) in zip(axes.flat, places, strict=False):
        local = table[(table.dataset == dataset) & (table.locale == locale)]
        pool = pools[(dataset, locale)]
        for share in sorted(SHARES):
            ranked = local[(local.share == share) & (local.rule.isin(["ranked"]))].copy()
            # The whole pool uses every sentence, so it closes both the ranked line and
            # the random line.
            if ranked.empty:
                continue
            ranked["x"] = [_x(b, pool) for b in ranked.budget]
            ranked = ranked.sort_values("x")
            ax.errorbar(
                ranked.x,
                ranked["mean"],
                yerr=ranked.sd.fillna(0),
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=colors(share / 100 * 0.9),
                label=f"{share}% human",
            )
        for share in (0, 100):
            random = local[(local.share == share) & (local.rule == "random")].copy()
            whole = local[(local.share == share) & (local.budget == "all")]
            random = pd.concat([random, whole])
            if random.empty:
                continue
            random["x"] = [_x(b, pool) for b in random.budget]
            random = random.sort_values("x")
            ax.plot(
                random.x,
                random["mean"],
                linestyle=":",
                color=colors(share / 100 * 0.9),
                linewidth=1,
                label=f"{share}% human, random" if ax is axes.flat[0] else None,
            )
        teacher = teacher_f1.get((dataset, locale))
        if teacher is not None:
            ax.axhline(teacher, color="red", linestyle="--", linewidth=1, label="teacher LLM")
        ax.set_xscale("log")
        ax.set_title(DATASETS[(dataset, locale)][0], fontsize="medium")
        ax.set_xlabel("Labelled sentences (log)", fontsize="small")
        ax.set_ylabel("Test micro-F1 (%)", fontsize="small")
        ax.tick_params(labelsize="x-small")
    for ax in list(axes.flat)[len(places) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize="x-small", ncol=1)
    return _save(fig, "mixing-curves", out)


def ranked_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Ranked fixed-N runs with `human` (human-labelled sentences) and `total` (N)."""
    cells = rows[(rows.rule == "ranked") & (rows.budget != "all")].copy()
    cells["total"] = cells.budget.astype(int)
    cells["human"] = (cells.share / 100 * cells.total).round().astype(int)
    return cells


def smallest_share(table: pd.DataFrame, teacher_f1: dict) -> dict:
    """{(dataset, locale, N): smallest human share (%) whose mean test F1 beats the teacher}.

    Descriptive: it reads the test grid and chooses nothing. The dev-chosen count is
    minimum.choose on ranked_cells.
    """
    out = {}
    ranked = table[(table.rule == "ranked") & (table.budget != "all")]
    for (dataset, locale, budget), cell in ranked.groupby(["dataset", "locale", "budget"]):
        teacher = teacher_f1.get((dataset, locale))
        if teacher is None:
            continue
        above = cell[cell["mean"] > teacher]
        out[(dataset, locale, int(budget))] = int(above.share.min()) if len(above) else None
    return out


def random_macros(table: pd.DataFrame) -> dict:
    """Random selection with teacher labels only (0%) or human labels only (100%)."""
    # Random runs also exist at 50 and 200 sentences (thesis budgets).
    names = {**BUDGETS, 50: "Fifty", 200: "TwoHundred"}
    out = {}
    for row in table[table.rule == "random"].to_dict("records"):
        key = (row["dataset"], row["locale"])
        budget = names.get(row["budget"])
        if key not in DATASETS or budget is None or row["share"] not in (0, 100):
            continue
        name = DATASETS[key][1] + "Curve" + SHARES[row["share"]] + "At" + budget + "Random"
        out[name] = row["mean"]
    return out


def fewest_macros(chosen: dict) -> dict:
    """The dev-chosen fewest human labels in the ranked grid, confirmed once on test."""
    out = {}
    for key, result in chosen.items():
        if key not in DATASETS or "test" not in result:
            continue
        name = DATASETS[key][1] + "Fewest"
        out[name + "Human"] = result["human"]
        out[name + "Teacher"] = result["total"] - result["human"]
        out[name + "Share"] = 100 * result["share"]
        out[name + "Fone"] = result.get("f1")
        out[name + "TestGap"] = result["test"]["gap"]
        out[name + "TestLow"] = result["test"]["low"]
        out[name + "TestHigh"] = result["test"]["high"]
    return {name: value for name, value in out.items() if value is not None}


def latex_table(table: pd.DataFrame, teacher_f1: dict, chosen: dict, path) -> Path:
    """References, smallest share that beats the teacher per N, and the dev-chosen cell."""
    lines = [
        "\\begin{tabular}{lrrr" + "r" * len(BUDGETS) + "rr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Whole pool} & \\multicolumn{"
        + str(len(BUDGETS))
        + "}{c}{Human share to beat the LLM} & \\multicolumn{2}{c}{Fewest (dev)} \\\\",
        "Dataset & LLM & LLM & Human & "
        + " & ".join(f"{n:,}".replace(",", "{,}") for n in BUDGETS)
        + " & Human+LLM & F1 \\\\",
        "\\midrule",
    ]
    shares = smallest_share(table, teacher_f1)
    for key, (label, _) in DATASETS.items():
        local = table[(table.dataset == key[0]) & (table.locale == key[1])]
        if local.empty:
            continue

        def whole(share, local=local):
            cell = local[(local.budget == "all") & (local.share == share)]
            return f"{cell['mean'].iloc[0]:.1f}" if len(cell) else "--"

        teacher = teacher_f1.get(key)
        cells = [label, f"{teacher:.1f}" if teacher is not None else "--", whole(0), whole(100)]
        for budget in BUDGETS:
            if (*key, budget) not in shares:
                cells.append("--")
            else:
                share = shares[(*key, budget)]
                cells.append("none" if share is None else f"{share}\\%")
        result = chosen.get(key, {})
        if "test" in result:
            cells.append(f"{result['human']}+{result['total'] - result['human']}")
            cells.append(f"{result['f1']:.1f}" if result.get("f1") is not None else "--")
        else:
            cells += ["--", "--"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
