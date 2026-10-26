"""The frozen 309-run design and its resumable runner."""

import hashlib
import json
import math
import time
import traceback
from itertools import product
from pathlib import Path

import yaml

from active_gliner import protocol, tasks
from active_gliner.observe.logs import append_json
from active_gliner.process import run_child
from active_gliner.run import RunConfig, experiment_dir
from active_gliner.teachers.config import REPO_ROOT

TRAINING_FIELDS = {
    "max_steps",
    "eval_steps",
    "batch_size",
    "grad_accum",
    "task_lr",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
    "augmentation",
    "relation_neg_ratio",
    "relation_head",
    "early_stopping_patience",
}


def recipe(dataset: str, *, required: bool = False) -> dict:
    task = tasks.for_dataset(dataset).__name__.rsplit(".", 1)[-1]
    path = REPO_ROOT / "configs/recipes" / f"{task}.yaml"
    if not path.exists():
        if required:
            raise ValueError(
                f"Missing frozen recipe: {path}; --allow-default-recipe is for smoke tests only"
            )
        return {}
    raw = path.read_bytes()
    values = yaml.safe_load(raw)
    if not isinstance(values, dict) or values.keys() - TRAINING_FIELDS:
        raise ValueError(f"Invalid training recipe: {path}")
    if required and dataset == "crossre" and "relation_neg_ratio" not in values:
        raise ValueError("The frozen CrossRE recipe must include relation_neg_ratio")
    if required and dataset == "crossre" and "relation_head" not in values:
        raise ValueError("The frozen CrossRE recipe must include relation_head")
    values = {**values, **MEMORY_SPLIT.get(dataset, {})}
    return {**values, "recipe_file": str(path), "recipe_hash": hashlib.sha256(raw).hexdigest()}


# BoolQ passages reach 823 words, and a batch of 8 of them needs more GPU memory than one
# run may take (23 Oct 2026). Two halves of 4 keep 8 examples per update, but it is not
# the same training: length grouping uses the physical batch, and batch 8 drops the last
# partial batch of each epoch while 4 x 2 trains it in a short extra update. Every BoolQ
# run uses this setup, so BoolQ arms stay comparable with each other.
MEMORY_SPLIT = {"boolq": {"batch_size": 4, "grad_accum": 2}}


# Training variants of the thesis replication; they override the frozen recipe.
VARIANTS = {
    "encoder-only": {"lora_targets": ["encoder"]},
    "heads-only": {"lora_targets": ["all_task_heads"]},
    "full-finetune": {"finetune": "full"},
}
MAIN_BUDGET = {"crossre": 200}
EQUIVALENCE_FILE = REPO_ROOT / "configs/protocol_equivalence.yaml"


def build() -> list[RunConfig]:
    runs = []
    ner = ["cleanconll", "bc5cdr", "mit_movie"]
    other = {"crossre": [100, 200], "hallmarks": [100, 400], "massive": [100, 400]}
    labels = ["ground_truth", "gemma-4-12b"]

    def add(
        block,
        datasets,
        selectors=("min", "random"),
        sources=labels,
        budgets=(100, 400),
        seeds=(1, 2, 3),
        **extra,
    ):
        for dataset, selector, source, n, seed in product(
            datasets, selectors, sources, budgets, seeds
        ):
            values = dict(
                dataset=dataset,
                selector=selector,
                labels_source=source,
                n=n,
                seed=seed,
                block=block,
                # Same seed, same scores: arm gaps never come from GPU rounding.
                exact_numerics=True,
            )
            values.update(recipe(dataset))
            values.update(extra)
            values.update(VARIANTS.get(extra.get("variant"), {}))
            runs.append(RunConfig(**values))

    add("ner_core", ner, selectors=("min", "random", "diversity"))
    add("ner_primary_extra_seeds", ner, budgets=(400,), seeds=(4, 5))
    for assignment in ("routed", "random"):
        add(
            "ner_routing",
            ner,
            selectors=("min",),
            sources=("gemma-4-12b",),
            budgets=(400,),
            gt_fraction=0.25,
            gt_assignment=assignment,
        )
    add("ner_qwen", ["cleanconll"], sources=("qwen-3.8-27b",))
    add(
        "ner_no_prediction_sensitivity",
        ["cleanconll"],
        selectors=("min",),
        budgets=(400,),
        no_prediction="last",
    )
    for dataset, budgets in other.items():
        add("other_tasks", [dataset], budgets=budgets)
        add("other_tasks_primary_extra_seeds", [dataset], budgets=(budgets[-1],), seeds=(4, 5))
    add("references", ner + list(other), selectors=("all",), budgets=(1,), seeds=(1,))
    add(
        "teacher_ladder_b",
        ["cleanconll", "hallmarks"],
        selectors=("random",),
        sources=("gemma-4-e4b", "gpt-oss-120b"),
        budgets=(400,),
    )
    add(
        "teacher_ladder_b",
        ["hallmarks"],
        selectors=("random",),
        sources=("qwen-3.8-27b",),
        budgets=(400,),
    )
    for fraction in (0.5, 0.75):
        add(
            "mixing",
            ["mit_movie"],
            selectors=("min",),
            sources=("gemma-4-12b",),
            budgets=(400,),
            gt_fraction=fraction,
            gt_assignment="random",
        )
    add("french", ["massive"], budgets=(400,), locale="fr-FR")
    add_teacher_family(add)
    add_thesis_replication(add)
    taken = {run_name(cfg) for cfg in runs}

    def add_new(*args, **kwargs):
        # The curves reuse every cell another block already runs.
        start = len(runs)
        add(*args, **kwargs)
        fresh = [cfg for cfg in runs[start:] if run_name(cfg) not in taken]
        runs[start:] = fresh
        taken.update(run_name(cfg) for cfg in fresh)

    add_mixing_curves(add_new)
    add_teacher_family_curves(add_new)
    add_decision_curves(add)
    add_open_ner_curves(add)
    return runs


# SST-5 is held back (18 Oct 2026): its licence is unknown and its five levels are binned
# crowd sliders. Its loader, splits and teacher labels stay on disk; it gets no runs.
DECISION_DATASETS = ("boolq", "clinc150")
# Open general-domain NER next to CleanCoNLL, whose Reuters text is not open.
OPEN_NER_DATASETS = ("uner_ewt",)


def add_decision_curves(add) -> None:
    """Decision tasks: the ranked mixing curves with both teachers, random lines, references.

    Design: docs/research/2026-10-18-1345-decision-tasks-design.md. Every input gets an
    answer, so "empty last" changes nothing; it keeps the run names of the other curves.
    The whole pool runs only as the references (all human, all Gemma, all Luna).
    """
    for dataset in DECISION_DATASETS:
        add_two_teacher_curves(add, "decision_curve", dataset)


def add_open_ner_curves(add) -> None:
    """UNER EWT gets the same cells as a decision dataset, under its own block name."""
    for dataset in OPEN_NER_DATASETS:
        add_two_teacher_curves(add, "open_ner_curve", dataset)


def add_two_teacher_curves(add, block: str, dataset: str) -> None:
    teachers = ("gemma-4-12b", "gpt-6-luna")
    budgets = tuple(n for n in CURVE_BUDGETS if n < 0.9 * pool_size(dataset, "en-US"))
    ranked = dict(selectors=("min",), no_prediction="last", budgets=budgets)
    add(block, [dataset], sources=("ground_truth", *teachers), **ranked)
    for share in CURVE_SHARES:
        add(
            block,
            [dataset],
            sources=teachers,
            gt_fraction=share,
            gt_assignment="random",
            **ranked,
        )
    add(
        block,
        [dataset],
        selectors=("random",),
        sources=("ground_truth", "gemma-4-12b"),
        budgets=budgets,
    )
    add(
        block,
        [dataset],
        selectors=("random",),
        sources=("gpt-6-luna",),
        budgets=(400,),
    )
    add(
        block,
        [dataset],
        selectors=("all",),
        sources=("ground_truth", *teachers),
        budgets=(1,),
        variant="long",
        max_steps=long_steps(dataset, "en-US"),
    )


def add_teacher_family_curves(add) -> None:
    """The ranked mixing curves again with GPT-6 Luna as the teacher.

    Checks that a small human share also beats a teacher from a second model family.
    Only the cells with teacher labels run: the 100% human cells and the random-selection
    lines do not depend on the teacher and are shared with the Gemma curves.
    """
    english = ["cleanconll", "bc5cdr", "mit_movie", "crossre", "hallmarks", "massive"]
    places = [(dataset, "en-US") for dataset in english] + [("massive", "fr-FR")]
    teacher = "gpt-6-luna"
    for dataset, locale in places:
        pool = pool_size(dataset, locale)
        budgets = tuple(n for n in CURVE_BUDGETS if n < 0.9 * pool)
        ranked = dict(selectors=("min",), no_prediction="last", budgets=budgets, locale=locale)
        whole = dict(
            selectors=("all",),
            budgets=(1,),
            locale=locale,
            variant="long",
            max_steps=long_steps(dataset, locale),
        )
        for cells in (ranked, whole):
            add("teacher_family_curve", [dataset], sources=(teacher,), **cells)
            for share in CURVE_SHARES:
                add(
                    "teacher_family_curve",
                    [dataset],
                    sources=(teacher,),
                    gt_fraction=share,
                    gt_assignment="random",
                    **cells,
                )


def add_teacher_family(add) -> None:
    """A teacher from a second model family (OpenAI GPT-6 Luna) on the main-table cells.

    Replaces the Cerebras API teachers, which never ran. Same selectors, budget and seeds
    as the Gemma cells, so the primary interaction can be read for a second teacher.
    """
    english = ["cleanconll", "bc5cdr", "mit_movie", "crossre", "hallmarks", "massive"]
    places = [(dataset, "en-US") for dataset in english] + [("massive", "fr-FR")]
    for dataset, locale in places:
        add(
            "teacher_family",
            [dataset],
            sources=("gpt-6-luna",),
            budgets=(MAIN_BUDGET.get(dataset, 400),),
            locale=locale,
        )


def add_thesis_replication(add) -> None:
    """Thesis experiments on every dataset (docs/research/2026-09-29-1047-*)."""
    english = ["cleanconll", "bc5cdr", "mit_movie", "crossre", "hallmarks", "massive"]
    places = [(dataset, "en-US") for dataset in english] + [("massive", "fr-FR")]
    for dataset, locale in places:
        main = MAIN_BUDGET.get(dataset, 400)
        # E7: thesis selectors.
        add(
            "thesis_selectors",
            [dataset],
            selectors=("avg", "mse", "mnlp"),
            sources=("ground_truth",),
            budgets=(100, main),
            locale=locale,
        )
        add(
            "thesis_selectors",
            [dataset],
            selectors=("avg", "mse", "mnlp"),
            sources=("gemma-4-12b",),
            budgets=(main,),
            locale=locale,
        )
        # E7: more budgets for min and random.
        extra = (50, 400, 1000) if dataset == "crossre" else (50, 200, 1000)
        add(
            "thesis_budgets",
            [dataset],
            sources=("ground_truth",),
            budgets=extra,
            locale=locale,
        )
        # E10: mixing grid, except cells that already exist.
        for fraction in (0.25, 0.5, 0.75):
            done = dataset == "mit_movie" or (
                fraction == 0.25 and dataset in {"cleanconll", "bc5cdr"}
            )
            if done:
                continue
            add(
                "thesis_mixing",
                [dataset],
                selectors=("min",),
                sources=("gemma-4-12b",),
                budgets=(main,),
                gt_fraction=fraction,
                gt_assignment="random",
                locale=locale,
            )
        # CrossRE's frozen recipe fixes its LoRA targets and relation head.
        if dataset == "crossre":
            continue
        # E4: LoRA layer groups.
        for variant in ("encoder-only", "heads-only"):
            add(
                "thesis_lora_layers",
                [dataset],
                selectors=("random",),
                sources=("ground_truth",),
                budgets=(main,),
                variant=variant,
                locale=locale,
            )
        # E2: full fine-tune on the whole pool.
        add(
            "thesis_full_finetune",
            [dataset],
            selectors=("all",),
            sources=("ground_truth",),
            budgets=(1,),
            seeds=(1,),
            variant="full-finetune",
            locale=locale,
        )


CURVE_BUDGETS = (100, 400, 1000, 2500)
CURVE_SHARES = (0.25, 0.5, 0.75)
MINIMUM_HUMAN_LABELS = (25, 50, 100, 200, 400, 1000, 2500)


def pool_size(dataset: str, locale: str) -> int:
    splits = json.loads((REPO_ROOT / "data/splits" / f"{dataset}-{locale}.json").read_text())
    return len(splits["pool"])


def long_steps(dataset: str, locale: str) -> int:
    """Steps for whole-pool curve runs: at least two passes over the pool.

    The default 1000 steps cover under one pass of a large pool, which would flatten
    the right end of the curve for a reason that has nothing to do with the labels.
    """
    values = recipe(dataset)
    batch = values.get("batch_size", RunConfig.model_fields["batch_size"].default)
    # One step learns from batch x grad_accum examples (MEMORY_SPLIT keeps this at 8).
    batch *= values.get("grad_accum", 1)
    return max(1000, math.ceil(2 * pool_size(dataset, locale) / batch))


def add_mixing_curves(add) -> None:
    """Thesis Fig 4.6.1 on every dataset, plus the minimum-human-labels curve.

    Design: docs/research/2026-10-02-1207-minimum-human-labels-design.md. The curves
    select by student confidence, as the thesis did, with no-prediction sentences
    last: the pool keeps sentences without entities (rule 4), and min would otherwise
    spend small budgets on them. Random selection gives the comparison lines for 0%
    and 100% human labels. The same N sentences serve every human share.
    """
    english = ["cleanconll", "bc5cdr", "mit_movie", "crossre", "hallmarks", "massive"]
    places = [(dataset, "en-US") for dataset in english] + [("massive", "fr-FR")]
    teacher = "gemma-4-12b"
    for dataset, locale in places:
        pool = pool_size(dataset, locale)
        budgets = tuple(n for n in CURVE_BUDGETS if n < 0.9 * pool)
        ranked = dict(selectors=("min",), no_prediction="last", budgets=budgets, locale=locale)
        # Curve 1: human share r at budget N; 0% = teacher only, 100% = human only.
        add("mixing_curve", [dataset], sources=("ground_truth", teacher), **ranked)
        for share in CURVE_SHARES:
            add(
                "mixing_curve",
                [dataset],
                sources=(teacher,),
                gt_fraction=share,
                gt_assignment="random",
                **ranked,
            )
        add(
            "mixing_curve",
            [dataset],
            selectors=("random",),
            sources=("ground_truth", teacher),
            budgets=budgets,
            locale=locale,
        )
        # The whole pool, with enough steps to see it.
        whole = dict(
            selectors=("all",),
            budgets=(1,),
            locale=locale,
            variant="long",
            max_steps=long_steps(dataset, locale),
        )
        add("mixing_curve", [dataset], sources=("ground_truth", teacher), **whole)
        for share in CURVE_SHARES:
            add(
                "mixing_curve",
                [dataset],
                sources=(teacher,),
                gt_fraction=share,
                gt_assignment="random",
                **whole,
            )
        # Curve 2: teacher labels on the whole pool, H of them replaced by human labels.
        # Nested: a larger human set extends the smaller one for the same seed.
        for human in MINIMUM_HUMAN_LABELS:
            if human >= pool:
                continue
            add(
                "minimum_human_labels",
                [dataset],
                sources=(teacher,),
                gt_fraction=human / pool,
                gt_assignment="nested",
                **whole,
            )
            # Two-step: start from the teacher-only whole-pool student of the same seed,
            # then train on the same H human sentences only. Step 2 uses the frozen
            # recipe, like the human-only runs it is compared with.
            add(
                "two_step",
                [dataset],
                selectors=("all",),
                sources=(teacher,),
                budgets=(1,),
                gt_fraction=human / pool,
                gt_assignment="nested",
                variant="two-step",
                locale=locale,
            )


def run_dir(cfg: RunConfig, out_root="runs") -> Path:
    return experiment_dir(cfg, out_root)


def run_name(cfg: RunConfig) -> str:
    return "/".join(run_dir(cfg).parts[1:])


def resolve(cfg: RunConfig, out_root="runs", *, required: bool = False) -> RunConfig:
    """Apply the frozen recipe with full validation.

    model_copy(update=...) skips validation, so an int recipe value (lora_alpha: 64)
    stays an int and changes the protocol fingerprint of an identical run.
    """
    values = {**cfg.model_dump(), **recipe(cfg.dataset, required=required)}
    values.update(VARIANTS.get(cfg.variant, {}))
    return RunConfig.model_validate({**values, "out_root": str(out_root)})


def equivalent_codes() -> list[str]:
    """Older code fingerprints whose runs were checked to reproduce under the current code."""
    if not EQUIVALENCE_FILE.exists():
        return []
    entries = yaml.safe_load(EQUIVALENCE_FILE.read_text()) or []
    return [entry["code"] for entry in entries if entry.get("checked") is True]


def run_status(cfg: RunConfig, out_root="runs") -> str:
    path = run_dir(cfg, out_root) / "metrics.json"
    if not path.exists():
        return "pending"
    try:
        stored = json.loads(path.read_text()).get("protocol_fingerprint")
    except (json.JSONDecodeError, AttributeError):
        return "stale"
    if not stored:
        return "stale"
    resolved = resolve(cfg, out_root)
    try:
        if stored == protocol.current_protocol(resolved):
            return "done"
        # A run made by checked older code still counts (configs/protocol_equivalence.yaml).
        for code in equivalent_codes():
            if stored == protocol.current_protocol(resolved, code):
                return "done"
    except FileNotFoundError:
        return "stale"
    return "stale"


def pending(runs: list[RunConfig], out_root="runs") -> list[RunConfig]:
    """Return unfinished runs; stale completed runs require explicit opt-in."""
    return [cfg for cfg in runs if not (run_dir(cfg, out_root) / "metrics.json").exists()]


def select(
    runs: list[RunConfig], blocks=None, datasets=None, labels_sources=None, locales=None
) -> list[RunConfig]:
    return [
        cfg
        for cfg in runs
        if (blocks is None or cfg.block in blocks)
        and (datasets is None or cfg.dataset in datasets)
        and (labels_sources is None or cfg.labels_source in labels_sources)
        and (locales is None or cfg.locale in locales)
    ]


def execute(
    runs: list[RunConfig],
    out_root="runs",
    limit: int | None = None,
    *,
    allow_default_recipe: bool = False,
    rerun_stale: bool = False,
) -> None:
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    resolved = [resolve(cfg, out_root, required=not allow_default_recipe) for cfg in runs]
    statuses = [(cfg, run_status(cfg, out_root)) for cfg in resolved]
    stale = [run_name(cfg) for cfg, status in statuses if status == "stale"]
    if stale and not rerun_stale:
        raise ValueError("Stale runs require --rerun-stale: " + ", ".join(stale))
    for cfg in [cfg for cfg, status in statuses if status != "done"][:limit]:
        directory = run_dir(cfg, out_root)
        started = time.monotonic()
        status = "completed"
        try:
            # Preserve an incomplete attempt before retrying in a clean run folder.
            if directory.exists():
                root = Path(out_root)
                archive = root.parent / f"{root.name}-archives" / directory.relative_to(root)
                archive = archive.with_name(archive.name + f"-attempt-{time.time_ns()}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                directory.rename(archive)
            code, stderr = run_child("active_gliner.matrix_run", [], stdin=cfg.model_dump_json())
            if code != 0:
                raise RuntimeError(f"Child exit code: {code}\n{stderr}")
        except Exception:
            status = "failed"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        append_json(
            Path(out_root) / "matrix_log.jsonl",
            {
                "name": run_name(cfg),
                "status": status,
                "wall_time_s": time.monotonic() - started,
            },
        )
