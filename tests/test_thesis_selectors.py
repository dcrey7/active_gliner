"""Thesis replication: avg, MSE and MNLP selectors, late config fields, checked old code."""

import math

import pytest
import yaml

from active_gliner import matrix, protocol, selection
from active_gliner.run import RunConfig


def test_scores_follow_the_thesis_formulas():
    # Same formulas as src2/active_gliner/selection/strategy.py. Its docstring examples
    # are miscomputed (0.0058, 0.2233, 0.1054); these are the values the code gives.
    assert selection.score_mse([0.9, 0.85, 0.95]) == pytest.approx(0.035 / 3)
    assert selection.score_mse([0.5, 0.6, 0.4]) == pytest.approx(0.77 / 3)
    expected = -(math.log(0.9) + math.log(0.85) + math.log(0.95)) / 3
    assert selection.score_mnlp([0.9, 0.85, 0.95]) == pytest.approx(expected)
    assert selection.score_avg([0.9, 0.8, 0.7]) == pytest.approx(0.8)


def test_no_prediction_follows_the_thesis_and_ranks_first():
    assert selection.score_avg([]) == 0.0
    assert selection.score_mse([]) == 1.0
    assert selection.score_mnlp([]) == math.inf
    for strategy in ("avg", "mse", "mnlp"):
        keys = selection.thesis_keys(strategy, [[0.9, 0.95], [], [0.4, 0.5]])
        picked = selection.select(["sure", "empty", "unsure"], keys, 2, strategy, seed=0)
        assert picked == ["empty", "unsure"]


def test_item_confidences_per_task_shape():
    assert selection.item_confidences({"spans": [{"confidence": 0.7}]}) == [0.7]
    assert selection.item_confidences({"relations": [{"probability": 0.6}], "pairs": []}) == [0.6]
    pred = {"probabilities": {"a": 0.8, "b": 0.2}, "labels": ["a"]}
    assert selection.item_confidences(pred) == [0.8]


def test_unknown_selector_still_fails():
    with pytest.raises(ValueError):
        selection.select(["a"], [0.1], 1, "entropy", seed=0)


def test_late_fields_at_default_leave_the_fingerprint_unchanged():
    base = RunConfig(dataset="cleanconll", n=400, seed=1)
    before = protocol.protocol_fingerprint(base, "pool", "prompt", code="old")
    same = RunConfig(dataset="cleanconll", n=400, seed=1, finetune="lora", variant=None)
    assert protocol.protocol_fingerprint(same, "pool", "prompt", code="old") == before
    full = RunConfig(dataset="cleanconll", n=400, seed=1, finetune="full")
    assert protocol.protocol_fingerprint(full, "pool", "prompt", code="old") != before


def test_only_checked_old_code_counts(tmp_path, monkeypatch):
    path = tmp_path / "equivalence.yaml"
    path.write_text(yaml.safe_dump([{"code": "aaa", "checked": True}, {"code": "bbb"}]))
    monkeypatch.setattr(matrix, "EQUIVALENCE_FILE", path)
    assert matrix.equivalent_codes() == ["aaa"]


def test_items_file_sits_next_to_the_pool_scores(tmp_path):
    score = tmp_path / "cleanconll-en-US-abc.jsonl"
    assert protocol.pool_items_path(score) == tmp_path / "cleanconll-en-US-abc-items.jsonl"
