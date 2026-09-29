"""Bring your own data: `active-gliner fit` on a user's JSONL file.

The community entry point (direction doc section 4): score, select, label with an
OpenAI-compatible teacher, LoRA train, report. Unit tests use a fake teacher.
"""

import json

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
