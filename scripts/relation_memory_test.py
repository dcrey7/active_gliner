"""Memorize 16 pool sentences; this is a diagnostic, not a dev/test score."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from active_gliner import data, model, pins, tasks
from active_gliner.observe.logs import write_json
from active_gliner.run import RunConfig
from active_gliner.tasks import relations
from active_gliner.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relation-head", choices=["classifier", "native", "marker"], required=True
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = RunConfig(
        dataset="crossre",
        n=16,
        seed=args.seed,
        relation_head=args.relation_head,
        max_steps=args.steps,
        eval_steps=args.steps,
        task_lr=args.lr,
        relation_neg_ratio=args.neg_ratio,
        batch_size=args.batch_size,
        augmentation=False,
        lora_r=16,
        lora_alpha=32,
        lora_targets={
            "classifier": ["encoder", "classifier"],
            "native": ["encoder", "relation_scorer"],
            "marker": ["encoder"],
        }[args.relation_head],
    )
    splits = data.load("crossre")
    labels = tasks.labels("crossre", splits)
    selected = splits["pool"][:16]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "selected_ids.json", [r.id for r in selected])
    write_json(args.out / "config.json", cfg.model_dump())
    revisions = {"crossre": data.dataset_pin("crossre")}
    pins.write_pins(
        args.out / "pins.json",
        pins.collect_pins(model.STUDENT_REPO, model.STUDENT_SHA, revisions),
    )
    seeds = {"data_order": args.seed * 1000 + 2, "lora_init": args.seed * 1000 + 3}
    student = model.load_student(device=args.device, local_files_only=True)
    result = train(student, cfg, seeds, args.out, relations, labels, selected, selected)
    del student
    model.free_gpu()
    student = model.load_adapter(
        model.load_student(device=args.device, local_files_only=True), args.out / "adapter/best"
    )
    student.eval()
    with torch.inference_mode():
        predictions = relations.predict(
            student, selected, labels, threshold=0.5, batch_size=args.batch_size
        )
    true, other = [], []
    for record, prediction in zip(selected, predictions, strict=True):
        gold = relations.gold_items(record)
        for pair in prediction["pairs"]:
            for label, probability in pair["probabilities"].items():
                target = true if (pair["head"], pair["tail"], label) in gold else other
                target.append(probability)
    output = {
        "purpose": "train_on_pool_evaluate_same_pool_memory_diagnostic",
        "relation_head": args.relation_head,
        "sentences": len(selected),
        "threshold": 0.5,
        **relations.score(selected, predictions, labels)["micro"],
        "median_true_probability": median(true) if true else None,
        "median_other_probability": median(other) if other else None,
        "training": result,
    }
    write_json(args.out / "memory_result.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
