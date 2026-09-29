"""Step 5 (O6): teacher prompts, validators, cache, client and stats.

Unit tests use a fake HTTP server (httpx.MockTransport). The last test,
marked `teacher`, calls the live local Gemma server if it is running.
"""

import json

import httpx
import pytest

from active_gliner import data, tasks
from active_gliner.data.records import Record
from active_gliner.teachers import TeacherConfig, label_records, prompts, validate
from active_gliner.teachers.cache import LabelCache


def _rec(text, task="ner", rid="t:en-US:1"):
    return Record(
        id=rid, doc_id=None, text=text, task=task, locale="en-US", source_split="train", gold={}
    )


NER_LABELS = {"PER": "person", "LOC": "location"}


# ---------- validators ----------


def test_ner_mentions_map_to_every_occurrence():
    r = _rec("Paris loves Paris .")
    content = json.dumps({"entities": [{"text": "Paris", "label": "location"}]})
    gold, errors = validate.parse(r, "ner", content, NER_LABELS)
    assert [(s["start"], s["end"], s["label"]) for s in gold["spans"]] == [
        (0, 5, "LOC"),
        (12, 17, "LOC"),
    ]
    assert errors == []


def test_ner_mention_skips_matches_inside_words():
    # Real MIT Movie case: the rating "r" must not land inside "are" or "rated".
    r = _rec("what movies are rated r")
    content = json.dumps({"entities": [{"text": "r", "label": "location"}]})
    gold, errors = validate.parse(r, "ner", content, NER_LABELS)
    assert [(s["start"], s["end"]) for s in gold["spans"]] == [(22, 23)]
    assert errors == []


def test_ner_mention_only_inside_a_word_is_an_error():
    # Real CleanCoNLL case: the student reads "Pakistan-ruled" as one word.
    r = _rec("a ravine in Pakistan-ruled Kashmir .")
    content = json.dumps(
        {
            "entities": [
                {"text": "Pakistan", "label": "location"},
                {"text": "Kashmir", "label": "location"},
            ]
        }
    )
    gold, errors = validate.parse(r, "ner", content, NER_LABELS)
    assert [s["text"] for s in gold["spans"]] == ["Kashmir"]
    assert [e["kind"] for e in errors] == ["not_on_word_edges"]


def test_every_accepted_span_is_a_student_training_target():
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    from active_gliner.tasks._spans import exact_token_span, training_values

    r = _rec("rated r films in Pakistan-ruled Kashmir by steven spielbergs")
    content = json.dumps(
        {
            "entities": [
                {"text": "r", "label": "location"},
                {"text": "Pakistan", "label": "location"},
                {"text": "steven spielberg", "label": "person"},
                {"text": "Kashmir", "label": "location"},
            ]
        }
    )
    gold, _ = validate.parse(r, "ner", content, NER_LABELS)
    labelled = Record(**{**r.__dict__, "gold": gold})
    for values in training_values(labelled, NER_LABELS).values():
        for value in values:
            exact_token_span(value, WhitespaceTokenSplitter())


def test_ner_drops_bad_items_but_keeps_good_ones():
    r = _rec("John met Mary in Rome .")
    content = json.dumps(
        {
            "entities": [
                {"text": "John", "label": "person"},
                {"text": "Rome", "label": "city"},  # label not allowed
                {"text": "Berlin", "label": "location"},  # not in text
            ]
        }
    )
    gold, errors = validate.parse(r, "ner", content, NER_LABELS)
    assert [s["text"] for s in gold["spans"]] == ["John"]
    assert sorted(e["kind"] for e in errors) == ["label_not_allowed", "not_in_text"]


def test_ner_empty_answer_is_valid():
    gold, errors = validate.parse(_rec("Nothing here ."), "ner", '{"entities": []}', NER_LABELS)
    assert gold == {"spans": []}
    assert errors == []


def test_bad_json_is_invalid_and_code_fences_are_repaired():
    r = _rec("John runs .")
    gold, errors = validate.parse(r, "ner", "not json at all", NER_LABELS)
    assert gold is None
    assert errors[0]["kind"] == "json"
    fenced = '```json\n{"entities": [{"text": "John", "label": "person"}]}\n```'
    gold, errors = validate.parse(r, "ner", fenced, NER_LABELS)
    assert [s["text"] for s in gold["spans"]] == ["John"]


def test_overlapping_mentions_keep_the_first_listed():
    r = _rec("New York City is big .")
    content = json.dumps(
        {
            "entities": [
                {"text": "New York City", "label": "location"},
                {"text": "New York", "label": "location"},
            ]
        }
    )
    gold, _ = validate.parse(r, "ner", content, NER_LABELS)
    assert [s["text"] for s in gold["spans"]] == ["New York City"]


def _rel_record():
    # Design section 15: entity mentions are given; ids = positions in gold["entities"].
    r = _rec("Marie works for Acme .", task="relations")
    r.gold = {
        "entities": [
            {"start": 0, "end": 5, "text": "Marie", "label": "person"},
            {"start": 16, "end": 20, "text": "Acme", "label": "organisation"},
        ],
        "relations": [],
    }
    return r


def test_relations_validator_uses_mention_ids():
    r = _rel_record()
    labels = {"role": "role", "origin": "origin"}
    content = json.dumps(
        {
            "relations": [
                {"head": 0, "tail": 1, "types": ["role"]},
                {"head": 0, "tail": 7, "types": ["origin"]},  # no mention 7
                {"head": 1, "tail": 0, "types": ["flying"]},  # label not allowed
            ]
        }
    )
    gold, errors = validate.parse(r, "relations", content, labels)
    assert gold["entities"] == r.gold["entities"]  # entities are input, kept as given
    assert [(x["head_id"], x["tail_id"], x["type"]) for x in gold["relations"]] == [(0, 1, "role")]
    assert gold["relations"][0]["head"]["text"] == "Marie"
    assert sorted(e["kind"] for e in errors) == ["bad_mention", "label_not_allowed"]


def test_relations_prompt_lists_mentions_with_ids():
    r = _rel_record()
    msgs = prompts.messages("relations", r.text, {"role": "role"}, record=r)
    user = msgs[-1]["content"]
    assert "[0] Marie (person)" in user
    assert "[1] Acme (organisation)" in user


def test_prompt_v2_adds_definitions_and_changes_hash():
    defs = {"PER": "A person.", "LOC": "A place."}
    v1 = prompts.messages("ner", "John runs .", NER_LABELS)
    v2 = prompts.messages("ner", "John runs .", NER_LABELS, definitions=defs)
    assert "A person." in v2[-1]["content"]
    assert "A person." not in v1[-1]["content"]
    assert prompts.prompt_hash("ner", NER_LABELS) != prompts.prompt_hash(
        "ner", NER_LABELS, definitions=defs
    )


def test_classification_validator():
    r = _rec("Tumours grow new vessels .", task="classification")
    labels = {"angio": "inducing angiogenesis", "death": "resisting cell death"}
    content = json.dumps({"labels": ["inducing angiogenesis", "flying"]})
    gold, errors = validate.parse(r, "classification", content, labels)
    assert gold == {"labels": ["angio"]}
    assert errors[0]["kind"] == "label_not_allowed"


def test_slots_validator():
    r = _rec("wake me up at seven am", task="slots")
    labels = {"time": "time", "date": "date"}
    content = json.dumps({"slots": {"time": ["seven am"], "date": []}})
    gold, errors = validate.parse(r, "slots", content, labels)
    assert [(s["text"], s["label"]) for s in gold["spans"]] == [("seven am", "time")]
    assert errors == []


# ---------- prompts ----------


def test_prompt_hash_is_stable_and_label_sensitive():
    a = prompts.prompt_hash("ner", NER_LABELS)
    assert a == prompts.prompt_hash("ner", dict(NER_LABELS))
    assert a != prompts.prompt_hash("ner", {"PER": "person"})
    msgs = prompts.messages("ner", "John runs .", NER_LABELS)
    assert "John runs ." in msgs[-1]["content"]
    assert "person" in msgs[-1]["content"]
    schema = prompts.response_schema("ner", NER_LABELS)
    assert schema["type"] == "object"


# ---------- client, cache, stats with a fake server ----------


def _fake(calls, fail_first=0):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        calls.append(json.loads(request.content))
        if state["n"] <= fail_first:
            return httpx.Response(500, json={"error": "busy"})
        text = calls[-1]["messages"][-1]["content"]
        ents = [{"text": "John", "label": "person"}] if "John" in text else []
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"entities": ents})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(handler)


def _cfg(tmp_path, **kw):
    return TeacherConfig(
        name="fake",
        base_url="http://fake/v1",
        model="fake-model",
        workers=2,
        price_in_per_m=1.0,
        price_out_per_m=2.0,
        retry_wait_s=0.0,
        **kw,
    )


def test_label_records_cache_resume_and_stats(tmp_path):
    recs = [_rec(f"John runs {i} .", rid=f"t:en-US:{i}") for i in range(5)]
    calls = []
    cfg = _cfg(tmp_path)
    out = label_records(recs, "ner", NER_LABELS, cfg, cache_dir=tmp_path, transport=_fake(calls))
    assert len(calls) == 5
    assert set(out.labels) == {r.id for r in recs}
    assert all(v["gold"]["spans"][0]["text"] == "John" for v in out.labels.values())
    assert out.stats["requests"] == 5
    assert out.stats["prompt_tokens"] == 500
    assert out.stats["completion_tokens"] == 100
    assert out.stats["cost_usd"] == pytest.approx((500 * 1.0 + 100 * 2.0) / 1e6)
    assert out.stats["valid_rate"] == 1.0

    # second call: everything comes from the cache
    calls2 = []
    out2 = label_records(recs, "ner", NER_LABELS, cfg, cache_dir=tmp_path, transport=_fake(calls2))
    assert calls2 == []
    assert out2.labels == out.labels
    assert out2.stats["cache_hits"] == 5


def test_request_body_is_deterministic_and_disables_thinking(tmp_path):
    calls = []
    label_records(
        [_rec("John runs .")],
        "ner",
        NER_LABELS,
        _cfg(tmp_path),
        cache_dir=tmp_path,
        transport=_fake(calls),
    )
    body = calls[0]
    assert body["temperature"] == 0
    assert body["model"] == "fake-model"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"


def test_retries_on_server_error(tmp_path):
    calls = []
    out = label_records(
        [_rec("John runs .")],
        "ner",
        NER_LABELS,
        _cfg(tmp_path),
        cache_dir=tmp_path,
        transport=_fake(calls, fail_first=2),
    )
    assert len(calls) == 3
    assert out.stats["retries"] == 2
    assert out.labels["t:en-US:1"]["gold"]["spans"][0]["text"] == "John"


def test_cache_file_is_keyed_by_prompt_hash(tmp_path):
    cache = LabelCache.for_job(tmp_path, "fake", "ner", "cleanconll", "en-US", "abc123")
    assert "abc123" in str(cache.path)
    cache.put("x", {"gold": {"spans": []}})
    assert LabelCache.for_job(tmp_path, "fake", "ner", "cleanconll", "en-US", "abc123").get("x")


def test_teacher_records_replace_gold():
    r = _rec("John runs .")
    labelled = {
        "t:en-US:1": {"gold": {"spans": [{"start": 0, "end": 4, "label": "PER", "text": "John"}]}}
    }
    new = validate.teacher_records([r], labelled)
    assert new[0].gold["spans"][0]["text"] == "John"
    assert r.gold == {}  # the original record is not changed


# ---------- live local teacher ----------


@pytest.mark.teacher
def test_live_gemma_labels_cleanconll(tmp_path):
    try:
        httpx.get("http://127.0.0.1:8020/health", timeout=2).raise_for_status()
    except Exception:
        pytest.skip("local Gemma teacher not running on port 8020")
    splits = data.load("cleanconll")
    recs = splits["dev"][:32]
    labels = tasks.labels("cleanconll", splits)
    cfg = TeacherConfig.from_yaml("configs/teachers/gemma-4-12b.yaml")
    out = label_records(recs, "ner", labels, cfg, cache_dir=tmp_path)
    assert out.stats["valid_rate"] >= 0.9
    task = tasks.for_dataset("cleanconll")
    preds = [{"id": r.id, "spans": out.labels[r.id]["gold"]["spans"]} for r in recs]
    f1 = task.score(recs, [dict(p, confidence=1.0) for p in preds], labels)["micro"]["f1"]
    assert f1 > 0.4
