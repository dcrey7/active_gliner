"""BCE training for given-mention heads, with pair batches and dev-only stopping."""

import random
import time

import torch
from gliner2.training.lora_targets import _resolve_targets
from gliner2.training.trainer import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.nn.functional import binary_cross_entropy_with_logits

from active_gliner.observe.logs import append_json, write_json
from active_gliner.observe.resources import sample
from active_gliner.tasks import relations
from active_gliner.tasks.relation_heads import configure, pair_logits, save_metadata


def prediction_loss(records, predictions, labels) -> float:
    """Mean BCE over all evaluated pair-label cells, for the existing loss plot."""
    probabilities, targets = [], []
    for record, prediction in zip(records, predictions, strict=True):
        gold = relations.gold_items(record)
        for pair in prediction["pairs"]:
            for label in labels:
                probabilities.append(pair["probabilities"][label])
                targets.append(float((pair["head"], pair["tail"], label) in gold))
    if not probabilities:
        return 0.0
    return torch.nn.functional.binary_cross_entropy(
        torch.tensor(probabilities), torch.tensor(targets)
    ).item()


def train_relations(
    student, cfg, seeds, run_dir, labels, selected, dev, eval_callback=None
) -> dict:
    """Use all label targets for each retained pair; evaluate on every dev pair."""
    configure(student, cfg.relation_head, labels, seeds["lora_init"])
    examples, rows = [], []
    for record in selected:
        examples.extend(relations.to_training_example(record, labels))
        rows.extend((record, i, j) for i, j in relations.pairs(record))
    for example, row in zip(examples, rows, strict=True):
        example["pair_row"] = row
    kept, counts = relations.subsample_negatives(
        examples, cfg.relation_neg_ratio, seeds["data_order"]
    )
    rows = [example["pair_row"] for example in kept]
    if not rows:
        raise ValueError("Relation subsampling leaves no training pairs: no positive pairs")
    write_json(run_dir / "training_pairs.json", counts)
    targets = ["encoder", "relation_scorer"] if cfg.relation_head == "native" else ["encoder"]
    resolved = _resolve_targets(student, targets)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds["lora_init"])
        student = get_peft_model(
            student,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=resolved,
                bias="none",
                modules_to_save=["marker_head"] if cfg.relation_head == "marker" else None,
            ),
        )
    # Full precision parameters keep small CPU checks and Adam updates stable.
    student.float()
    parameters = [p for p in student.parameters() if p.requires_grad]
    write_json(
        run_dir / "training_config.json",
        {
            **cfg.model_dump(),
            "lora_targets": targets,
            "resolved_lora_targets": resolved,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "marker_parameters": (
                2 * student.hidden_size**2
                + student.hidden_size
                + student.hidden_size * len(labels)
                + len(labels)
            )
            if cfg.relation_head == "marker"
            else 0,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
        },
    )
    optimizer = torch.optim.AdamW(parameters, lr=cfg.task_lr, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, "linear", cfg.max_steps, int(cfg.max_steps * 0.1))
    device = next(student.parameters()).device
    rng = random.Random(seeds["data_order"])
    order, cursor = [], 0
    best_f1, best_step, stale = -1.0, 0, 0
    started = time.monotonic()
    gold = {id(record): relations.gold_items(record) for record in selected}
    torch.manual_seed(seeds["data_order"])
    for step in range(1, cfg.max_steps + 1):
        student.train()
        optimizer.zero_grad()
        loss_value = 0.0
        for _ in range(cfg.grad_accum):
            if cursor >= len(order):
                order = list(range(len(rows)))
                rng.shuffle(order)
                cursor = 0
            chunk = [rows[i] for i in order[cursor : cursor + cfg.batch_size]]
            cursor += len(chunk)
            expected = torch.tensor(
                [[(i, j, label) in gold[id(record)] for label in labels] for record, i, j in chunk],
                dtype=torch.float32,
                device=device,
            )
            # Exact numerics trains in fp32, like the gliner2 trainer path.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and not cfg.exact_numerics,
            ):
                logits = pair_logits(student, chunk, labels)
                loss = binary_cross_entropy_with_logits(logits.float(), expected)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite relation loss at step {step}")
            (loss / cfg.grad_accum).backward()
            loss_value += loss.detach().item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        append_json(
            run_dir / "train_log.jsonl",
            {
                "step": step,
                "loss": loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "time_s": time.monotonic() - started,
                **sample(),
            },
        )
        if step % cfg.eval_steps == 0 or step == cfg.max_steps:
            predictions = relations.predict(
                student, dev, labels, threshold=cfg.eval_threshold, batch_size=cfg.batch_size
            )
            f1 = relations.score(dev, predictions, labels)["micro"]["f1"]
            append_json(
                run_dir / "eval_log.jsonl",
                {
                    "step": step,
                    "dev_f1": f1,
                    "eval_loss": prediction_loss(dev, predictions, labels),
                    "time_s": time.monotonic() - started,
                },
            )
            if f1 > best_f1:
                best_f1, best_step, stale = f1, step, 0
                path = run_dir / "adapter/best"
                student.save_pretrained(path)
                save_metadata(student, path)
            else:
                stale += 1
            if eval_callback is not None:
                eval_callback(step, f1)
            if stale >= cfg.early_stopping_patience:
                break
    return {
        "training_pairs": counts,
        "best_step": best_step,
        "best_dev_f1": best_f1,
        "stop_reason": "early_stopping" if step < cfg.max_steps else "max_steps",
        "train_time_s": time.monotonic() - started,
        "total_steps": step,
    }
