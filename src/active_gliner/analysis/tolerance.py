"""How much LLM labelling each task tolerates, compared across all task families.

At N = 1,000 ranked inputs, for every dataset and teacher: the teacher's own test F1,
the all-human student F1, the points lost when the LLM labels 25, 50, 75 or 100% of the
inputs, the largest LLM share that loses at most SAFE_LOSS points, and the smallest human
share whose student beats the teacher. Descriptive.
"""

from pathlib import Path

import pandas as pd

N = 1000
# A loss the paper calls small: about the seed SD of most cells (Table "main").
SAFE_LOSS = 2.0
LLM_SHARES = (25, 50, 75, 100)
TASK_FAMILY = {
    "cleanconll": "NER",
    "bc5cdr": "NER",
    "mit_movie": "NER",
    "uner_ewt": "NER",
    "crossre": "Relations",
    "hallmarks": "Classification",
    "massive": "Slots",
    "boolq": "Decision",
    "clinc150": "Decision",
    "sst5": "Decision",
}


def rows(tables: dict, teacher_f1: dict, places: dict) -> pd.DataFrame:
    """tables and teacher_f1 are keyed by teacher name; tables are curves.summary output."""
    out = []
    for teacher, table in tables.items():
        ranked = table[(table.rule == "ranked") & (table.budget == N)]
        for key, (title, name) in places.items():
            cell = ranked[(ranked.dataset == key[0]) & (ranked.locale == key[1])]
            cell = cell.set_index("share")["mean"]
            llm = teacher_f1.get(teacher, {}).get(key)
            if any(100 - share not in cell for share in LLM_SHARES) or llm is None:
                continue
            beats = [share for share in sorted(cell.index) if cell[share] > llm]
            loss = {share: float(cell[100] - cell[100 - share]) for share in LLM_SHARES}
            # Walk up the LLM share and stop at the first loss above SAFE_LOSS.
            safe = 0
            for share in LLM_SHARES:
                if loss[share] > SAFE_LOSS:
                    break
                safe = share
            out.append(
                dict(
                    dataset=key[0],
                    locale=key[1],
                    title=title,
                    name=name,
                    family=TASK_FAMILY[key[0]],
                    teacher=teacher,
                    llm=llm,
                    human=float(cell[100]),
                    # Teacher test F1 over the F1 of the student trained on these 1,000
                    # inputs with human labels ("T/S"): not human annotation accuracy.
                    quality=100 * llm / float(cell[100]),
                    loss_quarter_llm=loss[25],
                    loss_half_llm=loss[50],
                    loss_three_quarters_llm=loss[75],
                    loss_all_llm=loss[100],
                    safe_llm_share=safe,
                    share_to_beat=beats[0] if beats else None,
                )
            )
    return pd.DataFrame(out)


TEACHER_TAGS = {"gemma-4-12b": "Gemma", "gpt-6-luna": "Luna"}


def macros(table: pd.DataFrame) -> dict:
    found = {}
    for row in table.to_dict("records"):
        prefix = row["name"] + TEACHER_TAGS[row["teacher"]] + "Tolerance"
        found[prefix + "Quality"] = row["quality"]
        found[prefix + "Loss"] = row["loss_quarter_llm"]
        found[prefix + "LossHalf"] = row["loss_half_llm"]
        found[prefix + "LossThreeQuarters"] = row["loss_three_quarters_llm"]
        found[prefix + "LossAll"] = row["loss_all_llm"]
        found[prefix + "SafeShare"] = int(row["safe_llm_share"])
        # No human share beats the teacher (pandas stores the None as NaN).
        if row["share_to_beat"] is not None and not pd.isna(row["share_to_beat"]):
            found[prefix + "Share"] = int(row["share_to_beat"])
    return found


def _share(value) -> str:
    return "none" if value is None or pd.isna(value) else f"{int(value)}\\%"


def latex_table(table: pd.DataFrame, path) -> Path:
    """One row per dataset and teacher: points lost at each LLM share, and the safe share."""
    lines = [
        "\\begin{tabular}{lllrrrrrrr}",
        "\\toprule",
        " & & & & \\multicolumn{4}{c}{Points lost when the LLM labels} & & \\\\",
        "\\cmidrule(lr){5-8}",
        "Task & Dataset & Teacher & T/S & 25\\% & 50\\% & 75\\% & 100\\% & Safe & Beat \\\\",
        "\\midrule",
    ]
    groups = [] if table.empty else table.groupby(["dataset", "locale"], sort=False)
    for _, group in groups:
        for row in group.to_dict("records"):
            losses = [row[k] for k in LOSS_KEYS]
            lines.append(
                " & ".join(
                    [row["family"], row["title"], TEACHER_TAGS[row["teacher"]]]
                    + [f"{row['quality']:.0f}\\%"]
                    + [f"{value:.1f}" for value in losses]
                    + [f"{int(row['safe_llm_share'])}\\%", _share(row["share_to_beat"])]
                )
                + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


LOSS_KEYS = ("loss_quarter_llm", "loss_half_llm", "loss_three_quarters_llm", "loss_all_llm")
