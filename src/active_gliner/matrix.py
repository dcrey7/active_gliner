"""The frozen 309-run design and its resumable runner."""

import hashlib
import json
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
    return {**values, "recipe_file": str(path), "recipe_hash": hashlib.sha256(raw).hexdigest()}


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
    add_thesis_replication(add)
    return runs


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
    runs: list[RunConfig], blocks=None, datasets=None, labels_sources=None
) -> list[RunConfig]:
    return [
        cfg
        for cfg in runs
        if (blocks is None or cfg.block in blocks)
        and (datasets is None or cfg.dataset in datasets)
        and (labels_sources is None or cfg.labels_source in labels_sources)
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
