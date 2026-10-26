"""Bring your own data: `active-gliner fit` on a user's JSONL file.

The community entry point (direction doc section 4): score, select, label with an
OpenAI-compatible teacher, LoRA train, report. Unit tests use a fake teacher.
"""

import json
from pathlib import Path

import httpx
import pytest

from active_gliner import byod


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_read_texts_jsonl_and_plain(tmp_path):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"text": "Aspirin 100 mg daily ."}, {"id": "x7", "text": "Take ibuprofen ."}])
    recs = byod.read_texts(p, task="ner")
    assert [r.id for r in recs] == ["user:0", "user:x7"]
    assert recs[1].text == "Take ibuprofen ."
    q = tmp_path / "texts.txt"
    q.write_text("First line .\n\nSecond line .\n")
    assert [r.text for r in byod.read_texts(q, task="ner")] == ["First line .", "Second line ."]


def test_read_dev_with_labels(tmp_path):
    p = tmp_path / "dev.jsonl"
    _write(p, [{"text": "Aspirin 100 mg .", "entities": [{"text": "Aspirin", "label": "drug"}]}])
    recs = byod.read_labelled(p, task="ner")
    assert recs[0].gold["spans"][0]["start"] == 0
    assert recs[0].gold["spans"][0]["label"] == "drug"


def test_labels_argument():
    assert byod.parse_labels("drug, dose") == {"drug": "drug", "dose": "dose"}
    assert byod.parse_labels("drug:a medicine,dose") == {"drug": "drug", "dose": "dose"}
    assert byod.parse_definitions("drug:a medicine,dose") == {"drug": "a medicine"}


def test_teacher_from_url():
    cfg = byod.teacher_from_args("http://localhost:8000/v1", model="my-model", api_key_env=None)
    assert cfg.base_url == "http://localhost:8000/v1" and cfg.model == "my-model"


def test_plan_without_gpu(tmp_path):
    # --dry-run: no model, no teacher; prints what would happen
    p = tmp_path / "texts.jsonl"
    _write(p, [{"text": f"Sentence {i} ."} for i in range(50)])
    plan = byod.plan(texts=p, task="ner", labels="drug,dose", n=10, select="random", seed=1)
    assert plan["pool_size"] == 50
    assert plan["n"] == 10
    assert plan["select"] == "random"


def test_read_decision_labels(tmp_path):
    p = tmp_path / "dev.jsonl"
    _write(p, [{"text": "is it raining ?", "label": "yes"}, {"text": "x", "labels": ["no"]}])
    assert [r.gold["labels"] for r in byod.read_labelled(p, task="decision")] == [["yes"], ["no"]]
    _write(p, [{"text": "is it raining ?", "labels": ["yes", "no"]}])
    with pytest.raises(ValueError, match="exactly one"):
        byod.read_labelled(p, task="decision")


def test_plan_select_all_takes_every_text(tmp_path):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"text": f"Sentence {i} ."} for i in range(7)])
    assert byod.plan(texts=p, task="ner", labels="x", n=None, select="all", seed=1)["n"] == 7


def test_record_to_row_round_trips_every_task(tmp_path):
    from active_gliner.data.records import Record

    span = {"start": 0, "end": 7, "text": "Aspirin", "label": "DRUG"}
    cases = [
        ("ner", {"spans": [span]}, {"DRUG": "drug"}),
        ("classification", {"labels": ["A"]}, {"A": "alpha"}),
        ("decision", {"labels": ["Y"]}, {"Y": "yes"}),
        (
            "relations",
            {
                "entities": [span, {"start": 8, "end": 12, "text": "pain", "label": "DIS"}],
                "relations": [{"head_id": 0, "tail_id": 1, "type": "TREATS"}],
            },
            {"TREATS": "treats"},
        ),
    ]
    for task, gold, names in cases:
        record = Record("r1", None, "Aspirin pain .", task, "en-US", "dev", gold)
        path = tmp_path / f"{task}.jsonl"
        _write(path, [byod.record_to_row(record, names)])
        back = byod.read_labelled(path, task)[0].gold
        if task == "relations":
            assert back["relations"][0]["type"] == "treats"
            assert len(byod.record_to_row(record, names, with_labels=False)["entities"]) == 2
        elif task == "ner":
            assert back["spans"][0]["label"] == "drug"
        else:
            assert back["labels"] == [next(iter(names.values()))]
        if task != "relations":
            assert set(byod.record_to_row(record, names, with_labels=False)) == {"id", "text"}


def test_split_for_humans_is_a_seeded_random_share():
    rows = [{"text": str(i)} for i in range(100)]
    humans, rest = byod.split_for_humans(rows, 0.25, seed=3)
    assert len(humans) == 25 and len(rest) == 75
    assert {r["text"] for r in humans}.isdisjoint(r["text"] for r in rest)
    assert humans == byod.split_for_humans(rows, 0.25, seed=3)[0]
    # The share is random, not the first rows of the ranking.
    assert humans != rows[:25]
    with pytest.raises(ValueError):
        byod.split_for_humans(rows, 1.5, seed=3)


@pytest.mark.model
def test_rank_then_fit_mixes_human_and_llm_labels(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs the GPU")

    def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        body = {
            "choices": [{"message": {"content": json.dumps({"label": "yes"})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        assert "human" not in text  # human-labelled texts never reach the LLM
        return httpx.Response(200, json=body)

    p = tmp_path / "texts.jsonl"
    _write(p, [{"text": f"is item {i} in stock ?"} for i in range(40)])
    ranked = byod.rank(p, "decision", "yes,no", n=20, human_share=0.25, out_dir=tmp_path / "r")
    humans = [json.loads(line) for line in (ranked / "for_humans.jsonl").read_text().splitlines()]
    assert len(humans) == 5
    labelled = tmp_path / "human.jsonl"
    _write(labelled, [{"text": f"{row['text']} human", "label": "no"} for row in humans])
    out = byod.fit(
        texts=ranked / "for_llm.jsonl",
        task="decision",
        labels="yes,no",
        teacher_url="http://fake/v1",
        teacher_model="fake",
        select="all",
        human=labelled,
        max_steps=4,
        eval_steps=2,
        out_dir=tmp_path / "out",
        transport=httpx.MockTransport(handler),
    )
    split = json.loads((out / "split_ids.json").read_text())
    assert sum(i.startswith("human:") for i in split["train"]) == 5
    assert (out / "adapter/best/adapter_config.json").exists()


@pytest.mark.model
def test_fit_end_to_end_with_fake_teacher(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs the GPU")

    def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        ents = [{"text": "aspirin", "label": "drug"}] if "aspirin" in text else []
        body = {
            "choices": [{"message": {"content": json.dumps({"entities": ents})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        return httpx.Response(200, json=body)

    p = tmp_path / "texts.jsonl"
    _write(p, [{"text": f"patient {i} took aspirin today ."} for i in range(40)])
    out = byod.fit(
        texts=p,
        task="ner",
        labels="drug",
        teacher_url="http://fake/v1",
        teacher_model="fake",
        n=16,
        select="min",
        seed=1,
        max_steps=10,
        eval_steps=5,
        out_dir=tmp_path / "out",
        transport=httpx.MockTransport(handler),
    )
    for name in [
        "adapter/best/adapter_config.json",
        "report.md",
        "selected_ids.json",
        "teacher_stats.json",
    ]:
        assert (out / name).exists(), name


def test_packaged_recipes_equal_the_paper_configs():
    # The tool ships a copy of configs/recipes; both must stay the same.
    root = Path(__file__).resolve().parents[1]
    for task in byod.TASKS:
        packaged = (byod.RECIPES / f"{task}.yaml").read_text(encoding="utf-8")
        paper = (root / "configs" / "recipes" / f"{task}.yaml").read_text(encoding="utf-8")
        assert packaged == paper, task
    assert byod.task_recipe("relations")["relation_head"] == "native"
