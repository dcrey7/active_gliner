"""The second teacher family: GPT-6 Luna on the main-table cells.

Luna labels only the selected sentences, so it has min and random runs at the main
budget of every place, on fewer seeds than Gemma. Every comparison here uses only
the seeds that all three label sources share. The numbers are descriptive and stay
outside the Holm families.
"""

from pathlib import Path

import pandas as pd

from .tables import DATASETS, DEFAULT_BUDGET, MAIN_BUDGET, MAIN_TEACHER

SECOND_TEACHER = "gpt-6-luna"
# Label source and its macro name, in table order.
SOURCES = {"ground_truth": "GroundTruth", MAIN_TEACHER: "Gemma", SECOND_TEACHER: "Luna"}
CELLS = [(label, selector) for label in SOURCES for selector in ("min", "random")]


def cells(df) -> pd.DataFrame:
    """Seed-mean student F1 per place and label source, on the seeds all sources share."""
    rows = df[df.labels.isin(SOURCES) & df.selector.isin(["min", "random"])]
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    if "no_prediction" in rows:
        rows = rows[rows.no_prediction.fillna("zero") == "zero"]
    budget = rows.dataset.map(MAIN_BUDGET).fillna(DEFAULT_BUDGET)
    rows = rows[rows.n == budget]
    output = []
    for (dataset, locale), group in rows.groupby(["dataset", "locale"]):
        if SECOND_TEACHER not in set(group.labels):
            continue
        if group.duplicated(["seed", "labels", "selector"]).any():
            raise ValueError(f"Ambiguous second-teacher cells for {dataset}/{locale}")
        table = group.pivot(index="seed", columns=["labels", "selector"], values="f1")
        table = table.reindex(columns=pd.MultiIndex.from_tuples(CELLS)).dropna()
        if table.empty:
            continue
        row = {"dataset": dataset, "locale": locale, "seeds": len(table)}
        for label, name in SOURCES.items():
            row[name + "Min"] = float(table[(label, "min")].mean())
            row[name + "Random"] = float(table[(label, "random")].mean())
            row[name + "Gap"] = row[name + "Min"] - row[name + "Random"]
        output.append(row)
    return pd.DataFrame(output)


def overlap(teacher_tables: dict, dataset: str, locale: str) -> float | None:
    """Jaccard overlap of the Gemma and Luna error sentences on the test split."""
    entry = teacher_tables.get(f"{dataset}/{locale}", {})
    return entry.get("error_overlap", {}).get(MAIN_TEACHER, {}).get(SECOND_TEACHER)


def teacher_f1(teacher_tables: dict, dataset: str, locale: str, label: str) -> float | None:
    entry = teacher_tables.get("scores", {}).get(f"{dataset}/{locale}/{label}")
    return entry["f1"] if entry else None


def macros(table: pd.DataFrame, teacher_tables: dict) -> dict:
    found = {}
    teacher_gaps, student_gaps, overlaps = [], [], []
    for row in table.to_dict("records"):
        key = (row["dataset"], row["locale"])
        if key not in DATASETS:
            continue
        name = DATASETS[key][1]
        found[name + "LunaMin"] = row["LunaMin"]
        found[name + "LunaRandom"] = row["LunaRandom"]
        luna = teacher_f1(teacher_tables, *key, SECOND_TEACHER)
        gemma = teacher_f1(teacher_tables, *key, MAIN_TEACHER)
        if luna is not None:
            found["Teacher" + name + "Luna"] = luna
        if luna is not None and gemma is not None:
            teacher_gaps.append(luna - gemma)
        student_gaps.append(row["LunaMin"] - row["GemmaMin"])
        shared = overlap(teacher_tables, *key)
        if shared is not None:
            found[name + "FamilyOverlap"] = shared
            overlaps.append(shared)
    if table.empty:
        return found
    found["LunaPlaces"] = len(table)
    found["LunaSeeds"] = int(table.seeds.min())
    for name in SOURCES.values():
        found[name + "GapMeanLunaSeeds"] = float(table[name + "Gap"].mean())
        found[name + "GapPositiveLunaSeeds"] = int((table[name + "Gap"] > 0).sum())
    for name in ("Gemma", "Luna"):
        effect = table[name + "Gap"] - table["GroundTruthGap"]
        found[name + "InteractionMeanLunaSeeds"] = float(effect.mean())
    if teacher_gaps:
        found["LunaTeacherMinusGemma"] = sum(teacher_gaps) / len(teacher_gaps)
        found["LunaTeacherBetterCount"] = sum(gap > 0 for gap in teacher_gaps)
    found["LunaStudentMinusGemma"] = sum(student_gaps) / len(student_gaps)
    if overlaps:
        found["FamilyOverlapLow"] = min(overlaps)
        found["FamilyOverlapHigh"] = max(overlaps)
    return found


def _number(value) -> str:
    if value is None:
        return "--"
    # A hyphen prints too short; use a real minus sign.
    return f"$-${-value:.1f}" if value < 0 else f"{value:.1f}"


def latex_table(table: pd.DataFrame, teacher_tables: dict, path) -> Path:
    """Teacher F1, error overlap, student F1 and the min-random gap for both teachers."""
    by_key = {(r["dataset"], r["locale"]): r for r in table.to_dict("records")}
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Teacher F1} & Error & \\multicolumn{2}{c}{Student, Min}"
        " & \\multicolumn{3}{c}{Min $-$ Random} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){5-6} \\cmidrule(lr){7-9}",
        "Dataset & Gemma & Luna & overlap & Gemma & Luna & Gemma & Luna & Human \\\\",
        "\\midrule",
    ]
    for key, (title, _) in DATASETS.items():
        row = by_key.get(key)
        if row is None:
            continue
        shared = overlap(teacher_tables, *key)
        cells_ = [
            title,
            _number(teacher_f1(teacher_tables, *key, MAIN_TEACHER)),
            _number(teacher_f1(teacher_tables, *key, SECOND_TEACHER)),
            "--" if shared is None else f"{shared:.2f}",
            _number(row["GemmaMin"]),
            _number(row["LunaMin"]),
            _number(row["GemmaGap"]),
            _number(row["LunaGap"]),
            _number(row["GroundTruthGap"]),
        ]
        lines.append(" & ".join(cells_) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
