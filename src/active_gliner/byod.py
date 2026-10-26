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

TASKS = ("ner", "relations", "classification", "slots", "decision")
# The tuned training settings of the paper, one file per task type. They are a copy of
# configs/recipes/ shipped inside the package; a test keeps the two copies equal.
RECIPES = Path(__file__).with_name("recipes")
SELECTORS = ("min", "random", "all")


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
        elif task == "decision":
            # One answer per text, given as "label" or as a one-item "labels" list.
            answer = row["labels"] if "labels" in row else [row["label"]]
            if len(answer) != 1:
                raise ValueError(f"A decision needs exactly one label: {row}")
            record.gold = {"labels": answer}
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


def plan(texts: str | Path, task: str, labels: str, n: int | None, select: str, seed: int) -> dict:
    records = read_texts(texts, task)
    if select == "all":
        # "all" sends every text in the file to the LLM, for example the rest of a ranked set.
        n = len(records)
    if n is None or not 0 < n <= len(records):
        raise ValueError("n must be positive and no larger than the pool")
    if select not in SELECTORS or seed < 0:
        raise ValueError("Use min, random or all selection and a nonnegative seed")
    return {
        "pool_size": len(records),
        "n": n,
        "select": select,
        "task": task,
        "labels": parse_labels(labels),
    }


def task_recipe(task: str) -> dict:
    """The paper's tuned settings for this task type (learning rate, LoRA, relation head)."""
    import yaml

    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return yaml.safe_load((RECIPES / f"{task}.yaml").read_text(encoding="utf-8"))


def load_student(task: str, labels: dict[str, str], device: str | None = None):
    """The pinned student, set up as in the paper: relations use the tuned relation head."""
    import torch

    from active_gliner import model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student = model.load_student(device=device)
    head = task_recipe(task).get("relation_head", "classifier")
    if task == "relations" and head != "classifier":
        from active_gliner.tasks.relation_heads import configure

        configure(student, head, labels)
    return student


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
    n: int | None = None,
    select: str = "min",
    seed: int = 1,
    dev: str | Path | None = None,
    human: str | Path | None = None,
    max_steps: int = 1000,
    eval_steps: int = 100,
    out_dir: str | Path = "active-gliner-out",
    api_key_env: str | None = None,
    transport=None,
    threshold: float = 0.5,
    batch_size: int | None = None,
    grad_accum: int | None = None,
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
    n = summary["n"]
    pool = _pool(texts, task)
    # Human-labelled texts join the training set as they are; the LLM never sees them.
    human_records = read_labelled(human, task) if human is not None else []
    for record in human_records:
        record.id = f"human:{record.id}"
        record.source_split = "train"
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
        # The paper's rule: texts where the student predicts nothing go last, not first.
        no_prediction="last",
        labels_source="custom",
        **{
            **task_recipe(task),
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            # A smaller batch with accumulation keeps the update size when the GPU is shared.
            **({"batch_size": batch_size} if batch_size else {}),
            **({"grad_accum": grad_accum} if grad_accum else {}),
        },
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
                "human": str(human) if human else None,
                "task": task,
                "labels": labels,
                "definitions": definitions,
            }
        ),
        encoding="utf-8",
    )
    student = load_student(task, labels, device="cuda")
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
    selected = human_records + selected
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


def record_to_row(record: Record, names: dict[str, str], with_labels: bool = True) -> dict:
    """One JSONL row in the format fit, rank and predict read; names maps label codes to names.

    Relation rows always keep their mentions, because the mentions are input, not labels.
    """
    row = {"id": record.id, "text": record.text}
    gold, task = record.gold, record.task

    def entity(span, label):
        return {"text": span["text"], "start": span["start"], "end": span["end"], "label": label}

    if task == "relations":
        row["entities"] = [entity(e, e["label"]) for e in gold["entities"]]
        if with_labels:
            row["relations"] = [
                {"head": r["head_id"], "tail": r["tail_id"], "type": names[r["type"]]}
                for r in gold["relations"]
            ]
    elif not with_labels:
        pass
    elif task in {"ner", "slots"}:
        row["entities"] = [entity(s, names[s["label"]]) for s in gold["spans"]]
    elif task == "classification":
        row["labels"] = [names[label] for label in gold["labels"]]
    elif task == "decision":
        row["label"] = names[gold["labels"][0]]
    else:
        raise ValueError(f"Unknown task: {task}")
    return row


def split_for_humans(rows: list[dict], human_share: float, seed: int) -> tuple[list, list]:
    """A random share of the ranked rows goes to humans, the rest to the LLM."""
    if not 0 <= human_share <= 1:
        raise ValueError("human_share must be between 0 and 1")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    to_humans = set(order[: round(len(rows) * human_share)])
    humans = [row for i, row in enumerate(rows) if i in to_humans]
    rest = [row for i, row in enumerate(rows) if i not in to_humans]
    return humans, rest


def rank(
    texts: str | Path,
    task: str,
    labels: str,
    n: int,
    human_share: float = 0.25,
    seed: int = 1,
    out_dir: str | Path = "active-gliner-rank",
) -> Path:
    """Rank the texts once, least sure first, and split the top n between humans and the LLM.

    Writes for_humans.jsonl (a random share of the top n: label these by hand) and
    for_llm.jsonl (the rest: pass it to fit with select="all" and human=...).
    """
    import torch

    from active_gliner.observe.logs import write_jsonl
    from active_gliner.run import RunConfig, score_pool

    summary = plan(texts, task, labels, n, "min", seed)
    split_for_humans([], human_share, seed)
    pool = _pool(texts, task)
    cfg = RunConfig(
        dataset="user",
        out_root=str(Path(out_dir).parent),
        n=n,
        seed=seed,
        selector="min",
        no_prediction="last",
        labels_source="custom",
    )
    seeds = {"selection": seed * 1000 + 1}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    student = load_student(task, summary["labels"])
    student.eval()
    selected_ids, _ = score_pool(student, cfg, seeds, out, _task(task), summary["labels"], pool)
    del student
    gc.collect()
    torch.cuda.empty_cache()
    rows = {r.id: row for r, (_, row) in zip(pool, _rows(texts), strict=True)}
    humans, rest = split_for_humans([rows[i] for i in selected_ids], human_share, seed)
    write_jsonl(out / "for_humans.jsonl", humans)
    write_jsonl(out / "for_llm.jsonl", rest)
    return out


def predict(adapter: str | Path, task: str, labels: str, texts: str | Path) -> list[dict]:
    from active_gliner import model

    records = _pool(texts, task)
    if not records:
        return []
    student = model.load_adapter(load_student(task, parse_labels(labels)), adapter)
    student.eval()
    return _task(task).predict(student, records, parse_labels(labels))
