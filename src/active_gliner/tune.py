"""Dev-only Optuna search with one process per trial."""

import json
import random
from dataclasses import asdict
from pathlib import Path

from active_gliner import data, model, pins, tasks
from active_gliner.observe.logs import write_json, write_jsonl
from active_gliner.process import run_child
from active_gliner.run import RunConfig

FINISHED = {"COMPLETE", "PRUNED", "FAIL"}
RELATION_HEADS = ("classifier", "native", "marker")


def trial_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def trials_remaining(states: list[str], target: int) -> int:
    return max(0, target - sum(state in FINISHED for state in states))


def failed_trials(rows: list[dict]) -> list[int]:
    recovered = {
        row["retry_of"]
        for row in rows
        if row.get("retry_of") is not None and row["state"] in {"COMPLETE", "PRUNED"}
    }
    return [
        row["number"] for row in rows if row["state"] == "FAIL" and row["number"] not in recovered
    ]


def trials_to_retry(rows: list[dict]) -> list[int]:
    retried = {row.get("retry_of") for row in rows}
    return [
        row["number"]
        for row in rows
        if row["state"] == "FAIL" and row.get("retry_of") is None and row["number"] not in retried
    ]


def same_settings(stored: dict, current: dict) -> bool:
    """Compare search settings; a config field added later counts only at its default."""
    stored = {**stored, "config": RunConfig(**stored["config"]).model_dump()}
    return stored == current


def open_study(directory: Path, dataset: str, seed: int, pruner_warmup_steps: int = 0):
    import optuna

    return optuna.create_study(
        storage=f"sqlite:///{directory.resolve() / 'study.db'}",
        study_name=dataset,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        # A new head starts from zero dev F1; a warmup keeps it from early pruning.
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=pruner_warmup_steps),
    )


def trial_rows(study) -> list[dict]:
    return [
        {
            "number": trial.number,
            "params": trial.params,
            "training_params": trial.user_attrs.get("training_params"),
            "dev_f1": [{"step": step, "f1": f1} for step, f1 in trial.intermediate_values.items()],
            "wall_time_s": trial.duration.total_seconds() if trial.duration else None,
            "state": trial.state.name,
            "error": trial.user_attrs.get("error"),
            "value": trial.value,
            "vendor_inspired": trial.user_attrs.get("vendor_inspired", False),
            "retry_of": trial.user_attrs.get("retry_of"),
        }
        for trial in study.trials
    ]


def write_results(study, directory: Path) -> dict:
    rows = trial_rows(study)
    best = study.best_trial if any(row["state"] == "COMPLETE" for row in rows) else None
    result = {
        "trials": rows,
        "best_params": best.user_attrs["training_params"] if best else None,
        "best_dev_f1": best.value if best else None,
        "best_trial": best.number if best else None,
    }
    write_jsonl(directory / "trials.jsonl", rows)
    write_json(directory / "best.json", result)
    return result


def suggest_params(trial, dataset: str, relation_heads=RELATION_HEADS) -> dict:
    head = "classification_head" if dataset in {"crossre", "hallmarks"} else "extractive_head"
    targets = {
        "encoder_all": ["encoder", "all_task_heads"],
        "heads": ["all_task_heads"],
        "encoder_task": ["encoder", head],
    }
    params = {
        "task_lr": trial.suggest_float("task_lr", 1e-5, 1e-3, log=True),
        "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
        "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.1]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
    }
    params["lora_alpha"] = 2 * params["lora_r"]
    if dataset == "crossre":
        params["relation_head"] = trial.suggest_categorical("relation_head", list(relation_heads))
        params["augmentation"] = False
        params["lora_targets"] = {
            "classifier": ["encoder", "classifier"],
            "native": ["encoder", "relation_scorer"],
            "marker": ["encoder"],
        }[params["relation_head"]]
        params["relation_neg_ratio"] = trial.suggest_categorical(
            "relation_neg_ratio", [0.5, 1.0, 2.0, 4.0]
        )
    else:
        params["lora_targets"] = targets[trial.suggest_categorical("lora_targets", list(targets))]
        params["augmentation"] = trial.suggest_categorical("augmentation", [True, False])
    trial.set_user_attr("training_params", params)
    return params


def run_search(
    dataset: str,
    n_trials: int = 20,
    train_n: int | None = None,
    max_steps: int = 200,
    eval_steps: int = 50,
    dev_limit: int | None = None,
    out_root="runs",
    seed: int = 0,
    retry_failed: bool = False,
    relation_heads=RELATION_HEADS,
    pruner_warmup_steps: int = 0,
) -> dict:
    if n_trials <= 0 or (dev_limit is not None and dev_limit <= 0):
        raise ValueError("n_trials and dev_limit must be positive")
    if not relation_heads or set(relation_heads) - set(RELATION_HEADS):
        raise ValueError(f"relation_heads must be a nonempty subset of {RELATION_HEADS}")
    if pruner_warmup_steps < 0:
        raise ValueError("pruner_warmup_steps must not be negative")
    train_n = train_n if train_n is not None else (200 if dataset == "crossre" else 400)
    base = RunConfig(
        dataset=dataset, n=train_n, seed=seed, max_steps=max_steps, eval_steps=eval_steps
    )
    directory = Path(out_root) / "tuning" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    # Only one parent may allocate trials or recover an interrupted search.
    import fcntl

    with (directory / "search.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        search = {
            "relation_heads": list(relation_heads),
            "pruner_warmup_steps": pruner_warmup_steps,
        }
        return _search(directory, base, dev_limit, n_trials, out_root, retry_failed, search)


def _search(directory, base, dev_limit, n_trials, out_root, retry_failed, search):
    import optuna

    dataset, seed = base.dataset, base.seed
    settings = {"config": base.model_dump(), "dev_limit": dev_limit, **search}
    inputs_path = directory / "inputs.json"
    if inputs_path.exists():
        inputs = json.loads(inputs_path.read_text())
        if not same_settings(inputs["settings"], settings):
            raise ValueError("Resume settings differ from the frozen tuning inputs")
    else:
        if (directory / "study.db").exists():
            raise ValueError("The existing study is missing its frozen inputs.json")
        splits = data.load(dataset)
        selected = random.Random(seed).sample(splits["pool"], base.n)
        dev = splits["dev"][:dev_limit]
        if not dev:
            raise ValueError("The dev split is empty")
        data.write_split_ids(splits, dataset, "en-US", Path(out_root) / "splits")
        write_json(directory / "selected_ids.json", [r.id for r in selected])
        revisions = {dataset: data.dataset_pin(dataset)}
        pins.write_pins(
            directory / "pins.json",
            pins.collect_pins(
                model.STUDENT_REPO,
                model.STUDENT_SHA,
                revisions,
                extra={"split_hash": data.split_hash(splits), "optuna_version": optuna.__version__},
            ),
        )
        write_json(
            inputs_path,
            {
                "settings": settings,
                "labels": tasks.labels(dataset, splits),
                "selected": [asdict(r) for r in selected],
                "dev": [asdict(r) for r in dev],
            },
        )
    study = open_study(directory, dataset, seed, search["pruner_warmup_steps"])
    if not study.trials:
        study.enqueue_trial(
            {
                "task_lr": 5e-4,
                "lora_r": 4,
                "lora_dropout": 0.0,
                "lora_targets": "encoder_all",
                "augmentation": True,
                "batch_size": 8,
            },
            user_attrs={"vendor_inspired": True},
        )
    for frozen in study.trials:
        if frozen.state == optuna.trial.TrialState.RUNNING:
            trial = optuna.trial.Trial(study, frozen._trial_id)
            trial.set_user_attr(
                "error", "Parent interrupted; child exit code unknown; stderr unavailable"
            )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
    if retry_failed:
        for number in trials_to_retry(trial_rows(study)):
            failed = study.trials[number]
            study.enqueue_trial(
                {**failed.system_attrs.get("fixed_params", {}), **failed.params},
                user_attrs={"retry_of": number},
            )
    write_results(study, directory)
    # Explicit retries run even when failed attempts already fill the trial budget.
    while trials_remaining([t.state.name for t in study.trials], n_trials) or any(
        t.state == optuna.trial.TrialState.WAITING and "retry_of" in t.user_attrs
        for t in study.trials
    ):
        trial = study.ask()
        try:
            suggest_params(trial, dataset, search["relation_heads"])
            code, stderr = run_child(
                "active_gliner.tune_trial", [str(directory.resolve()), str(trial.number)]
            )
            # Refresh SQLite: the parent's Trial object predates the child's writes.
            attrs = study.trials[trial.number].user_attrs
            if code != 0 or "outcome" not in attrs:
                trial.set_user_attr("error", f"Child exit code: {code}\n" + stderr)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            elif attrs["outcome"] == "PRUNED":
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, attrs["result"])
        except Exception as error:
            trial.set_user_attr("error", trial_error(error))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        write_results(study, directory)
    return write_results(study, directory)


def freeze(dataset: str, result: dict) -> Path:
    import yaml

    from active_gliner.teachers.config import REPO_ROOT

    failed = failed_trials(result["trials"])
    if failed:
        raise ValueError(f"Cannot freeze a search with failed trials: {failed}")
    if result["best_params"] is None:
        raise ValueError("No completed trial to freeze")
    if dataset == "crossre" and "relation_neg_ratio" not in result["best_params"]:
        raise ValueError("CrossRE requires a search with relation_neg_ratio before freezing")
    if dataset == "crossre" and "relation_head" not in result["best_params"]:
        raise ValueError("CrossRE requires a search with relation_head before freezing")
    task = tasks.for_dataset(dataset).__name__.rsplit(".", 1)[-1]
    path = REPO_ROOT / "configs/recipes" / f"{task}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result["best_params"]), encoding="utf-8")
    return path
