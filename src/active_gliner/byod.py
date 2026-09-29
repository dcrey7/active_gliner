"""Fit the pinned student on user texts and teacher labels."""

import gc
import importlib
import json
import math
import random
import time
from pathlib import Path
from urllib.parse import urlparse

from active_gliner.data.records import Record
from active_gliner.teachers import TeacherConfig
from active_gliner.teachers.validate import empty_gold


def _rows(path):
    path = Path(path)
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line) if path.suffix == ".jsonl" else {"text": line.strip()}
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"Line {index}: expected a text string")
            yield index, row


def read_texts(path: str | Path, task: str) -> list[Record]:
    records = [
        Record(
            f"user:{row.get('id', index)}",
            None,
            row["text"],
            task,
            "en-US",
            "pool",
            empty_gold(task),
        )
        for index, row in _rows(path)
    ]
    if len({r.id for r in records}) != len(records):
        raise ValueError("Record ids must be unique")
    return records


def _entities(text, entities):
    result = []
    for entity in entities:
        if "start" in entity or "end" in entity:
            start, end = entity["start"], entity["end"]
        else:
            mention = entity["text"]
            start = text.find(mention)
            end = start + len(mention)
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid entity span: {entity}")
        if "text" in entity and entity["text"] != text[start:end]:
            raise ValueError(f"Entity text does not match offsets: {entity}")
        result.append(
            {"start": start, "end": end, "text": text[start:end], "label": entity["label"]}
        )
    return result


def read_labelled(path: str | Path, task: str) -> list[Record]:
    records = read_texts(path, task)
    for record, (_, row) in zip(records, _rows(path), strict=True):
        record.source_split = "dev"
        if task == "classification":
            record.gold = {"labels": row["labels"]}
        else:
            entities = _entities(record.text, row["entities"])
            if task != "relations":
                record.gold = {"spans": entities}
                continue
            relations = []
            for relation in row.get("relations", []):
                head, tail = relation["head"], relation["tail"]
                if (
                    any(type(i) is not int or not 0 <= i < len(entities) for i in (head, tail))
                    or head == tail
                ):
                    raise ValueError(f"Invalid relation: {relation}")
                relations.append(
                    {
                        "head_id": head,
                        "tail_id": tail,
                        "head": entities[head],
                        "tail": entities[tail],
                        "type": relation["type"],
                    }
                )
            record.gold = {"entities": entities, "relations": relations}
    return records


def parse_labels(value: str) -> dict[str, str]:
    names = [part.partition(":")[0].strip() for part in value.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Labels must be nonempty and unique")
    return {name: name for name in names}


def parse_definitions(value: str) -> dict[str, str]:
    parse_labels(value)
    return {
        name.strip(): definition.strip()
        for part in value.split(",")
        for name, colon, definition in [part.partition(":")]
        if colon and definition.strip()
    }


def teacher_from_args(url: str, model: str, api_key_env: str | None) -> TeacherConfig:
    return TeacherConfig(
        name="custom",
        base_url=url,
        model=model,
        api_key_env=api_key_env,
        send_template_kwargs=urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"},
    )


def plan(texts: str | Path, task: str, labels: str, n: int, select: str, seed: int) -> dict:
    records = read_texts(texts, task)
    if not 0 < n <= len(records):
        raise ValueError("n must be positive and no larger than the pool")
    if select not in {"min", "random"} or seed < 0:
        raise ValueError("Use min or random selection and a nonnegative seed")
    return {
        "pool_size": len(records),
        "n": n,
        "select": select,
        "task": task,
        "labels": parse_labels(labels),
    }


def _task(name):
    empty_gold(name)
    return importlib.import_module(f"active_gliner.tasks.{name}")


def _pool(path, task):
    records = read_texts(path, task)
    if task == "relations":
        # Supplied mentions are inputs; relation labels must never enter acquisition.
        for record, (_, row) in zip(records, _rows(path), strict=True):
            record.gold["entities"] = _entities(record.text, row.get("entities", []))
    return records


def fit(
    texts: str | Path,
    task: str,
    labels: str,
    teacher_url: str,
    teacher_model: str,
    n: int,
    select: str = "min",
    seed: int = 1,
    dev: str | Path | None = None,
    max_steps: int = 500,
    eval_steps: int = 50,
    out_dir: str | Path = "active-gliner-out",
    api_key_env: str | None = None,
    transport=None,
    threshold: float = 0.5,
) -> Path:
    import torch
    import yaml

    from active_gliner import model, pins
    from active_gliner.observe.logs import write_json
    from active_gliner.run import RunConfig, _hash, finish_run, score_pool
    from active_gliner.teachers import label_records, prompts, validate
    from active_gliner.train import train

    started = time.monotonic()
    summary = plan(texts, task, labels, n, select, seed)
    pool = _pool(texts, task)
    dev_records = read_labelled(dev, task) if dev is not None else None
    if dev_records is not None and not dev_records:
        raise ValueError("The dev set is empty")
    if dev_records is None and n < 2:
        raise ValueError("At least two selected texts are needed for a teacher holdout")
    if task == "relations" and not any(len(r.gold["entities"]) >= 2 for r in pool):
        raise ValueError("Relation training requires supplied entities in the texts JSONL")
    cfg = RunConfig(
        dataset="user",
        out_root=str(Path(out_dir).parent),
        n=n,
        seed=seed,
        selector=select,
        labels_source="custom",
        max_steps=max_steps,
        eval_steps=eval_steps,
        acq_threshold=0.5,
        eval_threshold=threshold,
    )
    seeds = {
        name: seed * 1000 + i for i, name in enumerate(("selection", "data_order", "lora_init"), 1)
    }
    teacher = teacher_from_args(teacher_url, teacher_model, api_key_env)
    definitions = parse_definitions(labels)
    definitions = {**summary["labels"], **definitions} if definitions else None
    version = "v2" if definitions else "v1"
    labels = summary["labels"]
    task_module = _task(task)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **cfg.model_dump(),
                "seeds": seeds,
                "texts": str(texts),
                "dev": str(dev) if dev else None,
                "task": task,
                "labels": labels,
                "definitions": definitions,
            }
        ),
        encoding="utf-8",
    )
    student = model.load_student(device="cuda")
    student.eval()
    selected_ids, _ = score_pool(student, cfg, seeds, out, task_module, labels, pool)
    by_id = {r.id: r for r in pool}
    selected = [by_id[i] for i in selected_ids]
    labelled = label_records(
        selected,
        task,
        labels,
        teacher,
        out.parent / "label-cache" / _hash(teacher.model_dump()),
        transport=transport,
        dataset="user",
        definitions=definitions,
        prompt_version=version,
    )
    write_json(out / "teacher_stats.json", labelled.stats)
    selected = validate.teacher_records(selected, labelled.labels)
    metric = "dev"
    if dev_records is None:
        random.Random(seeds["data_order"]).shuffle(selected)
        count = max(1, math.ceil(len(selected) * 0.1))
        dev_records, selected = selected[:count], selected[count:]
        metric = "teacher agreement (not accuracy)"
    splits = {"pool": pool, "train": selected, "dev": dev_records}
    write_json(
        out / "split_ids.json", {name: [r.id for r in rows] for name, rows in splits.items()}
    )
    write_json(
        out / "hashes.json",
        {
            "split": _hash({name: [r.id for r in rows] for name, rows in splits.items()}),
            "schema": _hash(labels),
            "config": _hash(cfg.model_dump()),
        },
    )
    run_pins = pins.collect_pins(
        model.STUDENT_REPO,
        model.STUDENT_SHA,
        {},
        extra={
            "teacher": {
                **teacher.model_dump(),
                "prompt_version": version,
                "prompt_hash": prompts.prompt_hash(task, labels, definitions),
                "label_cache_fingerprint": _hash(labelled.labels),
            }
        },
    )
    pins.write_pins(out / "pins.json", run_pins)
    training = train(student, cfg, seeds, out, task_module, labels, selected, dev_records)
    del student
    gc.collect()
    torch.cuda.empty_cache()
    return finish_run(
        cfg,
        seeds,
        out,
        task_module,
        labels,
        training,
        {"dev": dev_records},
        n,
        started,
        {"dev": metric},
    )


def predict(adapter: str | Path, task: str, labels: str, texts: str | Path) -> list[dict]:
    import torch

    from active_gliner import model

    records = _pool(texts, task)
    if not records:
        return []
    student = model.load_adapter(
        model.load_student(device="cuda" if torch.cuda.is_available() else "cpu"), adapter
    )
    student.eval()
    return _task(task).predict(student, records, parse_labels(labels))
