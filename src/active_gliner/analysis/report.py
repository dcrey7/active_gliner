"""Build the available paper artifacts and list missing inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate, cost, deployment, figures, ladder, numbers, selector_audit, stats, teacher
from . import tables as main_tables


def format_p(p: float, n_boot: int | None) -> str:
    """The relation and the value, for "$p$ <macro>" in the paper.

    A bootstrap p of 0 only means smaller than one over the number of resamples.
    """
    if p == 0:
        return f"$<{1 / (n_boot or 1000):g}$"
    return f"$={p:.3f}$"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _teacher_tables(df, skipped):
    from active_gliner import data

    tables, bins, scores = {}, {}, {}
    if df.empty:
        return tables, bins
    for (dataset, locale), group in df.groupby(["dataset", "locale"]):
        errors = {}
        # Zero-shot reference rows use labels "none": no teacher, no training labels.
        for label in sorted(set(group.labels) - {"ground_truth", "none"}):
            key = f"{dataset}/{locale}/{label}"
            try:
                counts = teacher.teacher_scores(dataset, locale, label, "test")
                _, records, values = teacher.cached_labels(dataset, locale, label, "test")
                errors[label] = {
                    r.id
                    for r, c, value in zip(records, counts, values, strict=True)
                    if c[1] or c[2] or value["gold"] is None or value["errors"]
                }
                scores[key] = {
                    "f1": 100 * stats.micro_f1(counts),
                    "n": len(counts),
                    "sentence_error_rate": len(errors[label]) / len(counts),
                    "invalid_rate": sum(v["gold"] is None or bool(v["errors"]) for v in values)
                    / len(values),
                }
            except FileNotFoundError as exc:
                skipped.append(f"Teacher test table {key}: {exc}")
            paths = [Path(p) / "pool_scores.jsonl" for p in group.run_dir]
            score_path = next((path for path in paths if path.exists()), None)
            if score_path is None:
                skipped.append(f"Teacher bins {key}: no pool confidence file")
                continue
            try:
                counts = teacher.teacher_scores(dataset, locale, label, "pool")
                records = data.load(dataset, locale)["pool"]
                confidence = {
                    row["id"]: row["confidence"]
                    for row in (json.loads(line) for line in score_path.read_text().splitlines())
                }
                ids = [i for i, record in enumerate(records) if record.id in confidence]
                bins[key] = teacher.f1_by_bin([confidence[records[i].id] for i in ids], counts[ids])
            except FileNotFoundError as exc:
                skipped.append(f"Teacher bins {key}: {exc}")
        tables[f"{dataset}/{locale}"] = {
            "error_overlap": teacher.error_overlap(errors).to_dict(),
            "scope": "Full cached test split; descriptive, outside the Holm families",
        }
    tables["scores"] = scores
    tables["bins"] = {name: table.to_dict("records") for name, table in bins.items()}
    return tables, bins


def _deployment_points(df, skipped):
    required = {"inference_cost_1m", "latency_ms", "timing_protocol", "model"}
    if not required <= set(df):
        skipped.append(
            "Deployment figures: missing measured inference costs, latency or model/protocol"
        )
        return pd.DataFrame()
    rows = df.dropna(subset=sorted(required)).copy()
    rows.loc[rows.selector == "all", "n"] = -1
    keys = [
        "task",
        "locale",
        "model",
        "labels",
        "selector",
        "n",
        "gt_fraction",
        "gt_assignment",
        "no_prediction",
        "timing_protocol",
    ]
    output = []
    for values, group in rows.groupby(keys, dropna=False):
        if group.duplicated(["dataset", "seed"]).any():
            skipped.append(f"Deployment point {values}: duplicate dataset/seed")
            continue
        datasets = sorted(group.dataset.unique())
        common_seeds = set.intersection(*(set(part.seed) for _, part in group.groupby("dataset")))
        group = group[group.seed.isin(common_seeds)]
        if group.empty:
            continue
        pooled = group.groupby("seed")[["f1", "inference_cost_1m", "latency_ms"]].mean()
        row = dict(zip(keys, values, strict=True))
        row.update(
            coverage=",".join(datasets),
            f1=float(pooled.f1.mean()),
            sd=float(pooled.f1.std()) if len(pooled) > 1 else None,
            cost=float(pooled.inference_cost_1m.mean()),
            latency_ms=float(pooled.latency_ms.mean()),
        )
        row["name"] = f"{row['model']}, {row['selector']}, {row['labels']}, N {row['n']}"
        for key in ("params", "family", "zero_shot_name"):
            if key in group and group[key].nunique() == 1:
                row[key] = group[key].dropna().iloc[0]
        output.append(row)
    return pd.DataFrame(output)


def analyse(runs="runs", out="paper") -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figure_dir = out / "figures"
    skipped = []
    df = aggregate.collect(runs)
    finished_runs = len(df)
    # Training variants (LoRA layers, full fine-tune) share selector, N and seed with main
    # runs; keep them out of every main-run table and contrast.
    variants = df[df.variant.notna()] if "variant" in df else df.iloc[0:0]
    df = df[df.variant.isna()] if "variant" in df else df
    primary = stats.primary_report(df, runs)
    tables, bins = _teacher_tables(df, skipped)
    tables["ladder"] = {}
    for dataset in sorted(df[df.locale == "en-US"].dataset.unique()):
        try:
            tables["ladder"][dataset] = ladder.scores(dataset, list(main_tables.LADDER_TEACHERS))
        except FileNotFoundError as exc:
            skipped.append(f"Teacher ladder {dataset}: {exc}")
    costs = []
    for row in df.to_dict("records"):
        costs.append({"name": row["name"], **cost.run_cost(row)})
    equal_cost = {}
    df["labelling_cost_usd"] = cost.labelling_cost(df, runs)
    teacher_rows = df.labels.isin(["gemma-4-12b"]) & (df.selector != "all")
    if teacher_rows.any() and df.loc[teacher_rows, "labelling_cost_usd"].isna().all():
        skipped.append("Cost curves: no free-GPU teacher timing in runs/timing (run time-teacher)")
    for dataset in sorted(df.dataset.unique()):
        equal_cost[dataset] = cost.equal_cost_table(df, dataset).to_dict("records")
        figures.learning_curves(df, dataset, figure_dir)
        figures.cost_curves(df, dataset, figure_dir)
    figures.interaction_forest(primary, figure_dir)
    figures.teacher_bins_chart(bins, figure_dir)
    # Different tasks and locales have different calibration populations.
    for (dataset, locale), group in df.groupby(["dataset", "locale"]):
        figures.calibration_chart(group.run_dir, figure_dir / f"{dataset}-{locale}")
    points = _deployment_points(df, skipped)
    if points.empty:
        # Per-run inference costs are not logged; build pooled NER points from timings.
        points = deployment.points(df, tables.get("scores", {}), runs)
        if not points.empty:
            skipped[:] = [s for s in skipped if not s.startswith("Deployment figures:")]
            figures.pareto_chart(points, figure_dir / "deployment-ner")
            figures.quadrant_chart(points, figure_dir / "deployment-ner")
    elif not points.empty:
        for keys, group in points.groupby(["task", "locale", "coverage", "timing_protocol"]):
            directory = figure_dir / (
                "deployment-" + "-".join(str(k).replace("/", "_") for k in keys)
            )
            figures.pareto_chart(group, directory)
            figures.quadrant_chart(group, directory)
    result = _clean(
        dict(
            runs=df.to_dict("records"),
            primary=primary,
            teachers=tables,
            costs=costs,
            equal_cost=equal_cost,
            deployment_points=points.to_dict("records"),
            prices=cost.prices(),
            skipped=skipped,
        )
    )
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    macros = {"FinishedRuns": finished_runs}
    names = {
        "ner": "NER",
        "relations": "Relations",
        "classification": "Classification",
        "slots": "Slots",
    }
    for family, prefix in (("interactions", "Primary"), ("practical", "Practical")):
        for task, entry in primary[family].items():
            if "mean" not in entry:
                continue
            name = prefix + names[task]
            macros[name] = entry["mean"]
            macros[name + "Sd"] = entry["sd"]
            interval = entry.get("interval", {})
            for key, suffix in (("low", "Low"), ("high", "High")):
                if key in interval:
                    macros[name + suffix] = interval[key]
            if entry.get("p_holm") is not None:
                macros[name + "P"] = format_p(entry["p_holm"], entry["interval"].get("n_boot"))
    macros.update(main_tables.cell_macros(df))
    macros.update(main_tables.mixing_macros(df))
    selector_means = main_tables.selector_macros(df)
    macros.update(selector_means)
    macros.update(selector_audit.thesis_selector_macros(df, selector_means))
    macros.update(main_tables.threshold_macros(df))
    macros.update(main_tables.variant_macros(variants))
    macros.update(main_tables.teacher_macros(tables.get("scores", {}), tables.get("bins", {})))
    macros.update(main_tables.ladder_macros(tables["ladder"]))
    macros.update(deployment.macros(points))
    numbers.write_macros(out / "numbers.tex", macros)
    main_tables.write_main_table(df, out / "tables" / "main_results.tex")
    return result
