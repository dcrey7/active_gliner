"""Descriptive: is the teacher worse on sentences the zero-shot student is unsure about?

Pool only (never test). Changes no setting. Teacher labels come from the on-disk cache.
"""

import json
import random
import sys

from active_gliner import data, model, protocol, tasks
from active_gliner.teachers import TeacherConfig, label_records, prompts, validate
from active_gliner.teachers.config import REPO_ROOT

DATASETS = sys.argv[1:] or ["cleanconll", "bc5cdr", "mit_movie", "hallmarks", "massive"]


def micro_f1(gold_sets, pred_sets):
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets, strict=True))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets, strict=True))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets, strict=True))
    return round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 1.0


student = model.load_student(device="cuda")
teacher = TeacherConfig.from_yaml(REPO_ROOT / "configs/teachers/gemma-4-12b.yaml")
for name in DATASETS:
    splits = data.load(name, "en-US")
    task = tasks.for_dataset(name)
    labels = tasks.labels(name, splits)
    pool = splits["pool"]
    version = prompts.version_for(name)
    labelled = label_records(
        pool,
        pool[0].task,
        labels,
        teacher,
        REPO_ROOT / "labels",
        dataset=name,
        definitions=prompts.prompt_for(name, version),
        prompt_version=version,
    )
    teacher_gold = {
        r.id: task.gold_items(r) for r in validate.teacher_records(pool, labelled.labels)
    }
    with protocol.inference_precision():
        preds = task.predict(
            student, pool, labels, threshold=0.5, batch_size=32, slot_mode="structure"
        )
    student_items = {p["id"]: task.pred_items(p) for p in preds}
    conf = {p["id"]: p["confidence"] for p in preds}
    gold = {r.id: task.gold_items(r) for r in pool}
    order = sorted(pool, key=lambda r: conf[r.id])
    k = len(pool) // 10
    groups = {
        "least_sure_10pct": order[:k],
        "random_10pct": random.Random(0).sample(pool, k),
        "most_sure_10pct": order[-k:],
    }
    for decile in range(10):
        groups[f"decile_{decile + 1:02d}"] = order[decile * k : (decile + 1) * k]
    row = {"dataset": name, "pool": len(pool), "teacher_valid_rate": labelled.stats["valid_rate"]}
    for group, records in groups.items():
        ids = [r.id for r in records]
        row[group] = {
            "teacher_f1": micro_f1([gold[i] for i in ids], [teacher_gold[i] for i in ids]),
            "student_zero_shot_f1": micro_f1(
                [gold[i] for i in ids], [student_items[i] for i in ids]
            ),
            "gold_items_per_sentence": round(sum(len(gold[i]) for i in ids) / len(ids), 2),
            "empty_gold_share": round(sum(not gold[i] for i in ids) / len(ids), 3),
        }
    print(json.dumps(row), flush=True)
