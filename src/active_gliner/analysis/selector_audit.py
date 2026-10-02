"""Why the thesis selectors agree: shared selections and empty-prediction pools.

avg, mse and mnlp all rank sentences with no prediction first. When a pool holds more
such sentences than the budget, all three pick the same seeded subset of them.
"""

import json
from pathlib import Path

import pandas as pd

from .tables import DATASETS, LABELS

THESIS = ("avg", "mse", "mnlp")


def _selected(run_dir) -> frozenset:
    data = json.loads((Path(run_dir) / "selected_ids.json").read_text())
    ids = data["ids"] if isinstance(data, dict) else data
    return frozenset(map(str, ids))


def identity_macros(df) -> dict:
    """Count cells (dataset, labels, budget, seed) where avg, mse and mnlp chose the same set.

    NER cells also report whether min chose that set too.
    """
    rows = df[df.selector.isin((*THESIS, "min"))]
    if "gt_fraction" in rows:
        rows = rows[rows.gt_fraction.isna()]
    # The no-prediction sensitivity runs also use min; keep the default rule only.
    if "no_prediction" in rows:
        rows = rows[rows.no_prediction.fillna("zero") == "zero"]
    cells = identical = ner_cells = ner_with_min = 0
    for (_, _, _, task, _, _), group in rows.groupby(
        ["dataset", "locale", "labels", "task", "n", "seed"]
    ):
        by_selector = {row.selector: _selected(row.run_dir) for row in group.itertuples()}
        if not all(s in by_selector for s in THESIS):
            continue
        cells += 1
        same = len({by_selector[s] for s in THESIS}) == 1
        identical += same
        if task == "ner" and "min" in by_selector:
            ner_cells += 1
            ner_with_min += same and by_selector["min"] == by_selector["avg"]
    if not cells:
        return {}
    return {
        "ThesisSelectorCells": cells,
        "ThesisSelectorIdenticalCells": identical,
        "ThesisSelectorNERCells": ner_cells,
        "ThesisSelectorNERMinIdenticalCells": ner_with_min,
    }


def spread_macros(selector_macros: dict) -> dict:
    """Range of mean F1 across avg, mse and mnlp per dataset and labels (points)."""
    macros = {}
    for _, (_, dataset) in DATASETS.items():
        for _, (_, labels) in LABELS.items():
            values = [selector_macros.get(dataset + labels + s.capitalize()) for s in THESIS]
            if all(v is not None for v in values):
                macros[dataset + labels + "ThesisSpread"] = max(values) - min(values)
    return macros


def no_prediction_macros(df) -> dict:
    """Share (%) and count of pool sentences with no zero-shot prediction, per dataset."""
    macros = {}
    for (dataset, locale), group in df.groupby(["dataset", "locale"]):
        if (dataset, locale) not in DATASETS:
            continue
        rows = None
        for run_dir in group.run_dir:
            acquisition = Path(run_dir) / "acquisition.json"
            if not acquisition.exists():
                continue
            artifact = Path(json.loads(acquisition.read_text()).get("pool_score_artifact", ""))
            if artifact.is_file():
                rows = [json.loads(line) for line in artifact.read_text().splitlines() if line]
                break
        if not rows:
            continue
        empty = sum(not row.get("has_prediction", True) for row in rows)
        name = DATASETS[(dataset, locale)][1]
        macros[name + "NoPredictionCount"] = empty
        macros[name + "NoPredictionShare"] = 100 * empty / len(rows)
    return macros


def thesis_selector_macros(df: pd.DataFrame, selector_macros: dict) -> dict:
    macros = identity_macros(df)
    macros.update(spread_macros(selector_macros))
    macros.update(no_prediction_macros(df))
    return macros
