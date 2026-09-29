"""The main results table and per-cell macros, made from finished runs only.

One row per dataset and label source, at the main budget of that task. Cells show
mean test micro-F1 over seeds, the seed SD, and the number of seeds.
"""

from pathlib import Path

import pandas as pd

# Macro names must be ASCII letters only (numbers.write_macros).
DATASETS = {
    ("cleanconll", "en-US"): ("CleanCoNLL", "CleanCoNLL"),
    ("bc5cdr", "en-US"): ("BC5CDR", "BCfiveCDR"),
    ("mit_movie", "en-US"): ("MIT Movie", "MITMovie"),
    ("crossre", "en-US"): ("CrossRE", "CrossRE"),
    ("hallmarks", "en-US"): ("Hallmarks", "Hallmarks"),
    ("massive", "en-US"): ("MASSIVE en", "MassiveEn"),
    ("massive", "fr-FR"): ("MASSIVE fr", "MassiveFr"),
}
LABELS = {"ground_truth": ("Ground truth", "GroundTruth"), "gemma-4-12b": ("Gemma 4 12B", "Gemma")}
# The smaller ladder teacher: macros only, not a row of the main table.
LADDER_LABELS = {"gemma-4-e4b": ("Gemma 4 E4B", "EfourB")}
MAIN_TEACHER = "gemma-4-12b"
LADDER_TEACHERS = {"gemma-4-12b": "TwelveB", "gemma-4-e4b": "EfourB"}
# Column title and macro suffix, in table order.
ARMS = {
    "random": ("Random", "Random"),
    "min": ("Min", "Min"),
    "min_last": ("Min, empty last", "MinLast"),
    "diversity": ("Diversity", "Diversity"),
    "all": ("Whole pool", "WholePool"),
}
MAIN_BUDGET = {"crossre": 200}
DEFAULT_BUDGET = 400
ZERO_SHOT_STUDENT = "gliner2.5-multi-v1"


def _arm(row) -> str | None:
    if row["selector"] == "min" and row.get("no_prediction") == "last":
        return "min_last"
    return row["selector"] if row["selector"] in ARMS else None


def summary(df, labels=LABELS) -> pd.DataFrame:
    """Mean, SD and seed count per dataset, locale, label source and arm."""
    if df.empty:
        return pd.DataFrame(columns=["dataset", "locale", "labels", "arm", "mean", "sd", "seeds"])
    rows = df.copy()
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    rows = rows[rows.labels.isin(labels)]
    if "no_prediction" not in rows:
        rows["no_prediction"] = "zero"
    rows["no_prediction"] = rows.no_prediction.fillna("zero")
    rows["arm"] = [_arm(row) for row in rows.to_dict("records")]
    budget = rows.dataset.map(MAIN_BUDGET).fillna(DEFAULT_BUDGET)
    rows = rows[rows.arm.notna() & ((rows.arm == "all") | (rows.n == budget))]
    grouped = rows.groupby(["dataset", "locale", "labels", "arm"]).f1
    result = grouped.agg(mean="mean", sd="std", seeds="count").reset_index()
    return result


def zero_shot(df) -> pd.DataFrame:
    if "zero_shot_name" not in df:
        return pd.DataFrame(columns=["dataset", "locale", "f1"])
    rows = df[df.zero_shot_name == ZERO_SHOT_STUDENT]
    return rows[["dataset", "locale", "f1"]]


def cell_macros(df) -> dict:
    macros = {}
    for row in summary(df).to_dict("records"):
        key = (row["dataset"], row["locale"])
        if key in DATASETS and row["labels"] in LABELS:
            name = DATASETS[key][1] + LABELS[row["labels"]][1] + ARMS[row["arm"]][1]
            macros[name] = row["mean"]
    for row in zero_shot(df).to_dict("records"):
        key = (row["dataset"], row["locale"])
        if key in DATASETS:
            macros[DATASETS[key][1] + "ZeroShot"] = row["f1"]
    for row in summary(df, LADDER_LABELS).to_dict("records"):
        key = (row["dataset"], row["locale"])
        if key in DATASETS:
            name = DATASETS[key][1] + LADDER_LABELS[row["labels"]][1] + ARMS[row["arm"]][1]
            macros[name] = row["mean"]
    return macros


MIX_FRACTIONS = {0.25: "TwentyFive", 0.5: "Fifty", 0.75: "SeventyFive"}
MIX_ASSIGNMENTS = {"random": "", "routed": "Routed"}


def mixing_macros(df) -> dict:
    """Mean F1 of the mixing runs: a share of the selected sentences gets ground truth.

    Name: dataset + Mix + fraction (+ Routed when ground truth went to the least sure).
    """
    if "gt_fraction" not in df:
        return {}
    rows = df[df.gt_fraction.notna()]
    macros = {}
    keys = ["dataset", "locale", "selector", "gt_fraction", "gt_assignment"]
    for (dataset, locale, selector, fraction, assignment), group in rows.groupby(keys):
        if (dataset, locale) not in DATASETS or selector != "min":
            continue
        if fraction not in MIX_FRACTIONS or assignment not in MIX_ASSIGNMENTS:
            continue
        name = (
            DATASETS[(dataset, locale)][1]
            + "Mix"
            + MIX_FRACTIONS[fraction]
            + MIX_ASSIGNMENTS[assignment]
        )
        macros[name] = group.f1.mean()
    return macros


VARIANT_NAMES = {
    "encoder-only": "EncoderOnly",
    "heads-only": "HeadsOnly",
    "full-finetune": "FullFinetune",
}
THESIS_SELECTORS = {"avg": "Avg", "mse": "Mse", "mnlp": "Mnlp"}


def variant_macros(variants) -> dict:
    """Mean F1 of the thesis training variants per dataset (LoRA layers, full fine-tune)."""
    macros = {}
    for (dataset, locale, variant), group in variants.groupby(["dataset", "locale", "variant"]):
        if (dataset, locale) in DATASETS and variant in VARIANT_NAMES:
            macros[DATASETS[(dataset, locale)][1] + VARIANT_NAMES[variant]] = group.f1.mean()
    return macros


def selector_macros(df) -> dict:
    """Mean F1 of the thesis selectors at the main budget: dataset + labels + selector."""
    macros = {}
    rows = df[df.selector.isin(THESIS_SELECTORS)]
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    for (dataset, locale, labels, selector, n), group in rows.groupby(
        ["dataset", "locale", "labels", "selector", "n"]
    ):
        if (dataset, locale) not in DATASETS or labels not in LABELS:
            continue
        if n != MAIN_BUDGET.get(dataset, DEFAULT_BUDGET):
            continue
        name = DATASETS[(dataset, locale)][1] + LABELS[labels][1] + THESIS_SELECTORS[selector]
        macros[name] = group.f1.mean()
    return macros


def threshold_macros(df) -> dict:
    """Thesis E6: spread of dev F1 over thresholds 0.3 to 0.7, zero-shot against trained.

    Trained = ground truth, random, main budget, mean over seeds. Dev only, descriptive.
    """
    if "dev_threshold_spread" not in df:
        return {}
    macros = {}
    for (dataset, locale), group in df.groupby(["dataset", "locale"]):
        if (dataset, locale) not in DATASETS:
            continue
        name = DATASETS[(dataset, locale)][1]
        zero = group[group.get("zero_shot_name", pd.Series(dtype=str)) == ZERO_SHOT_STUDENT]
        zero = zero.dev_threshold_spread.dropna()
        if len(zero):
            macros[name + "ZeroShotSpread"] = zero.mean()
        budget = MAIN_BUDGET.get(dataset, DEFAULT_BUDGET)
        tuned = group[
            (group.labels == "ground_truth") & (group.selector == "random") & (group.n == budget)
        ].dev_threshold_spread.dropna()
        if len(tuned):
            macros[name + "TrainedSpread"] = tuned.mean()
    return macros


def teacher_macros(scores: dict, bins: dict) -> dict:
    """Main-teacher test F1, and its F1 in the least and most confident student bins."""
    macros = {}
    for (dataset, locale), (_, name) in DATASETS.items():
        key = f"{dataset}/{locale}/{MAIN_TEACHER}"
        if key in scores:
            macros["Teacher" + name] = scores[key]["f1"]
        if key in bins and len(bins[key]):
            table = pd.DataFrame(bins[key]).sort_values("bin")
            macros["Teacher" + name + "LeastSure"] = 100 * table.f1.iloc[0]
            macros["Teacher" + name + "MostSure"] = 100 * table.f1.iloc[-1]
    return macros


def ladder_macros(ladder: dict) -> dict:
    """Teacher ladder on the frozen English test samples (keys are dataset names)."""
    macros = {}
    for dataset, entry in ladder.items():
        name = "Ladder" + DATASETS[(dataset, "en-US")][1]
        for teacher, f1 in entry["f1"].items():
            macros[name + LADDER_TEACHERS[teacher]] = f1
        if "gap" in entry:
            macros[name + "Gap"] = entry["gap"]["mean"]
            macros[name + "GapLow"] = entry["gap"]["interval"]["low"]
            macros[name + "GapHigh"] = entry["gap"]["interval"]["high"]
        overlap = entry["error_overlap"].get("gemma-4-12b", {}).get("gemma-4-e4b")
        if overlap is not None:
            macros[name + "Overlap"] = overlap
        macros[name + "Sentences"] = entry["n"]
    return macros


def _cell(entry) -> str:
    if entry is None:
        return "--"
    if entry["seeds"] < 2 or pd.isna(entry["sd"]):
        return f"{entry['mean']:.1f}"
    return f"{entry['mean']:.1f} $\\pm$ {entry['sd']:.1f} ({entry['seeds']})"


def main_table(df) -> str:
    cells = {
        (r["dataset"], r["locale"], r["labels"], r["arm"]): r
        for r in summary(df).to_dict("records")
    }
    zero = {(r["dataset"], r["locale"]): r["f1"] for r in zero_shot(df).to_dict("records")}
    header = ["Dataset", "Labels", "Zero-shot"] + [title for title, _ in ARMS.values()]
    lines = [
        "\\begin{tabular}{ll" + "r" * (len(header) - 2) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for (dataset, locale), (title, _) in DATASETS.items():
        for label, (label_title, _) in LABELS.items():
            first = label == "ground_truth"
            zero_cell = f"{zero[(dataset, locale)]:.1f}" if (dataset, locale) in zero else "--"
            row = [title if first else "", label_title, zero_cell if first else ""]
            row += [_cell(cells.get((dataset, locale, label, arm))) for arm in ARMS]
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_main_table(df, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(main_table(df), encoding="utf-8")
    return path
