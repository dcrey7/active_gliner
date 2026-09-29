"""Execute one allocated SQLite trial, then release CUDA by exiting."""

import json
import math
import sys
from pathlib import Path

import optuna
import yaml

from active_gliner import data, model, tasks
from active_gliner.observe.logs import write_json
from active_gliner.run import RunConfig
from active_gliner.train import train
from active_gliner.tune import open_study


def main() -> None:
    directory = Path(sys.argv[1])
    number = int(sys.argv[2])
    inputs = json.loads((directory / "inputs.json").read_text())
    base = inputs["settings"]["config"]
    # The child must prune with the same rule as the parent that allocated the trial.
    warmup = inputs["settings"].get("pruner_warmup_steps", 0)
    study = open_study(directory, base["dataset"], base["seed"], warmup)
    frozen = study.trials[number]
    if frozen.state != optuna.trial.TrialState.RUNNING:
        raise ValueError(f"Trial {number} is not RUNNING")
    trial = optuna.trial.Trial(study, frozen._trial_id)
    cfg = RunConfig(**{**base, **trial.user_attrs["training_params"]})
    selected = [data.Record(**row) for row in inputs["selected"]]
    if [r.id for r in selected] != json.loads((directory / "selected_ids.json").read_text()):
        raise ValueError("Frozen selected ids differ from the saved records")
    dev = [data.Record(**row) for row in inputs["dev"]]
    trial_dir = directory / f"trial-{number}"
    trial_dir.mkdir()
    (trial_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    write_json(trial_dir / "pins.json", json.loads((directory / "pins.json").read_text()))

    def report(step, f1):
        trial.report(f1, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = train(
            model.load_student(device="cuda"),
            cfg,
            {"data_order": cfg.seed * 1000 + 2, "lora_init": cfg.seed * 1000 + 3},
            trial_dir,
            tasks.for_dataset(cfg.dataset),
            inputs["labels"],
            selected,
            dev,
            eval_callback=report,
        )
    except optuna.TrialPruned:
        trial.set_user_attr("outcome", "PRUNED")
    else:
        if not math.isfinite(result["best_dev_f1"]):
            raise ValueError("Trial returned a non-finite dev F1")
        trial.set_user_attr("result", result["best_dev_f1"])
        trial.set_user_attr("outcome", "COMPLETE")


if __name__ == "__main__":
    main()
