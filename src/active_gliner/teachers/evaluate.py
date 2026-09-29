"""Evaluate a fixed prompt on the first dev records of each dataset."""

import json
from pathlib import Path

from active_gliner import data, tasks

from .client import label_records
from .config import REPO_ROOT, TeacherConfig
from .prompts import prompt_for, prompt_hash
from .validate import teacher_records


def evaluate(teacher: str, prompt: str, n: int = 200) -> Path:
    if n <= 0 or prompt not in {"v1", "v2"}:
        raise ValueError("Use a positive n and prompt v1 or v2")
    cfg = TeacherConfig.from_yaml(REPO_ROOT / "configs/teachers" / f"{teacher}.yaml")
    path = Path("runs/teacher_eval") / f"{teacher}-{prompt}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for dataset in data.LOADERS:
            for locale in ("en-US", "fr-FR") if dataset == "massive" else ("en-US",):
                splits = data.load(dataset, locale)
                records = splits["dev"][:n]
                if not records:
                    raise ValueError(f"Empty dev split: {dataset}/{locale}")
                labels = tasks.labels(dataset, splits)
                task = tasks.for_dataset(dataset)
                definitions = prompt_for(dataset, prompt)
                result = label_records(
                    records,
                    records[0].task,
                    labels,
                    cfg,
                    REPO_ROOT / "labels",
                    dataset=dataset,
                    definitions=definitions,
                    prompt_version=prompt,
                )
                predictions = []
                for record in teacher_records(records, result.labels):
                    prediction = {"id": record.id, **record.gold}
                    if record.task == "relations":
                        prediction["relations"] = [
                            {"head": r["head_id"], "tail": r["tail_id"], "type": r["type"]}
                            for r in record.gold["relations"]
                        ]
                    predictions.append(prediction)
                scores = task.score(records, predictions, labels)
                row = {
                    "dataset": dataset,
                    "locale": locale,
                    "teacher": teacher,
                    "prompt": prompt,
                    "n": len(records),
                    "ids": [r.id for r in records],
                    "prompt_hash": prompt_hash(records[0].task, labels, definitions),
                    "micro_f1": scores["micro"]["f1"],
                    "macro_f1": scores["macro_f1"],
                    "valid_rate": result.stats["valid_rate"],
                    "errors_by_kind": result.stats["errors_by_kind"],
                    "scores": scores,
                    "stats": result.stats,
                }
                line = json.dumps(row, ensure_ascii=False)
                stream.write(line + "\n")
                stream.flush()
                print(line, flush=True)
    return path
