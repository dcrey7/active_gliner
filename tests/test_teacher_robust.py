"""The labelling client survives a crashed or flaky server (25 Sep 2026 crash)."""

import json

import httpx

from active_gliner.data.records import Record
from active_gliner.teachers import TeacherConfig, label_records

LABELS = {"PER": "person"}


def _rec(i):
    return Record(f"t:en-US:{i}", None, f"John runs {i} .", "ner", "en-US", "train", {})


def _cfg():
    return TeacherConfig(
        name="fake",
        base_url="http://fake/v1",
        model="fake",
        workers=1,
        max_retries=2,
        retry_wait_s=0.0,
        max_wait_server_s=0,
    )


def _ok():
    body = {"entities": [{"text": "John", "label": "person"}]}
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def test_connection_errors_are_retried(tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return _ok()

    out = label_records(
        [_rec(0)], "ner", LABELS, _cfg(), cache_dir=tmp_path, transport=httpx.MockTransport(handler)
    )
    assert out.labels["t:en-US:0"]["gold"]["spans"][0]["text"] == "John"
    assert out.stats["retries"] >= 1


def test_a_dead_server_fails_records_without_caching_or_crashing(tmp_path):
    def dead(request):
        raise httpx.ConnectError("[Errno 111] Connection refused")

    recs = [_rec(i) for i in range(3)]
    out = label_records(
        recs, "ner", LABELS, _cfg(), cache_dir=tmp_path, transport=httpx.MockTransport(dead)
    )
    assert out.stats["errors_by_kind"].get("request_failed") == 3
    # nothing cached: a rerun against a live server labels all three
    calls = []

    def live(request):
        calls.append(1)
        return _ok()

    out2 = label_records(
        recs, "ner", LABELS, _cfg(), cache_dir=tmp_path, transport=httpx.MockTransport(live)
    )
    assert len(calls) == 3
    assert len(out2.labels) == 3
