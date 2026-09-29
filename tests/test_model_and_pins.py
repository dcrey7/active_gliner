"""Step 1 (O3): pinned student loading and the pins file."""

import json
import os

import pytest

from active_gliner import model, pins


def test_student_constants_are_pinned():
    assert model.STUDENT_REPO == "fastino/gliner2.5-multi-v1"
    assert len(model.STUDENT_SHA) == 40


def test_collect_pins_has_all_fields():
    p = pins.collect_pins(
        model_repo=model.STUDENT_REPO,
        model_sha=model.STUDENT_SHA,
        dataset_revisions={"cleanconll": "abc123"},
    )
    assert p["model"] == {"repo": model.STUDENT_REPO, "sha": model.STUDENT_SHA}
    for name in ["gliner2", "torch", "transformers", "peft", "python"]:
        assert p["packages"][name]
    assert p["packages"]["gliner2"] == "2.0.0"
    assert set(p["cuda"]) >= {"available", "version", "device_name"}
    assert set(p["git"]) >= {"commit", "dirty"}
    assert p["datasets"] == {"cleanconll": "abc123"}
    assert p["created_at"]


def test_write_pins_roundtrip(tmp_path):
    p = pins.collect_pins(model.STUDENT_REPO, model.STUDENT_SHA, {})
    out = tmp_path / "pins.json"
    pins.write_pins(out, p)
    assert json.loads(out.read_text()) == p


def _snapshot_cached() -> bool:
    try:
        model.snapshot_path(model.STUDENT_REPO, model.STUDENT_SHA, local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.model
@pytest.mark.skipif(not _snapshot_cached(), reason="student snapshot not in the local HF cache")
def test_load_student_offline_from_snapshot(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    m1 = model.load_student(device="cpu", local_files_only=True)
    assert type(m1).__name__ == "BoundaryExtractor"
    out = m1.extract_entities(
        "Peter Blackburn met officials in Brussels.",
        ["person", "location"],
        include_confidence=True,
    )
    people = [e["text"] for e in out["entities"]["person"]]
    assert "Peter Blackburn" in people

    # Every call returns a new object: LoRA must never leak between runs.
    m2 = model.load_student(device="cpu", local_files_only=True)
    assert m1 is not m2


def test_snapshot_path_is_under_the_sha():
    if not _snapshot_cached():
        pytest.skip("student snapshot not in the local HF cache")
    path = model.snapshot_path(model.STUDENT_REPO, model.STUDENT_SHA, local_files_only=True)
    assert model.STUDENT_SHA in os.fspath(path)
