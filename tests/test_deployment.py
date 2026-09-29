"""Deployment points: pooled NER F1 with free-GPU timings (design, hero figures)."""

import json

import pandas as pd
import pytest

from active_gliner import timing
from active_gliner.analysis import cost, deployment


def _zero(name, dataset, f1, latency, params=287_000_000):
    return {
        "zero_shot_name": name,
        "dataset": dataset,
        "locale": "en-US",
        "labels": "none",
        "selector": "zero_shot",
        "n": None,
        "seed": None,
        "f1": f1,
        "latency_ms": latency,
        "params": params,
    }


def _run(dataset, labels, selector, seed, f1):
    return {
        "zero_shot_name": None,
        "dataset": dataset,
        "locale": "en-US",
        "labels": labels,
        "selector": selector,
        "n": 400,
        "seed": seed,
        "f1": f1,
        "latency_ms": None,
        "params": None,
        "gt_fraction": None,
        "no_prediction": "zero",
    }


def _df():
    rows = [_zero(deployment.STUDENT_BASE, d, 50.0, 2.0) for d in deployment.NER_DATASETS]
    rows += [_zero("small", d, 40.0, 1.0, 70_000_000) for d in deployment.NER_DATASETS]
    for dataset in deployment.NER_DATASETS:
        for seed, f1 in ((1, 80.0), (2, 82.0)):
            rows.append(_run(dataset, "ground_truth", "random", seed, f1))
    return pd.DataFrame(rows)


def test_zero_shot_points_pool_the_three_datasets():
    points = {p["name"]: p for p in deployment.zero_shot_points(_df())}
    assert points["small"]["f1"] == 40.0
    assert points["small"]["params"] == pytest.approx(0.07)
    rate = cost.prices()["gpu"]["usd_per_hour"]
    assert points["small"]["cost"] == pytest.approx(1.0 / 1000 * 1e6 / 3600 * rate)


def test_students_reuse_the_base_timing_and_average_seeds():
    [student] = deployment.student_points(_df())
    assert student["f1"] == 81.0
    assert student["sd"] == pytest.approx(1.414, abs=1e-3)
    assert student["latency_ms"] == 2.0
    assert student["zero_shot_name"] == deployment.STUDENT_BASE  # arrow start in the quadrant


def test_teacher_needs_scores_and_timing_for_every_dataset(tmp_path):
    scores = {f"{d}/en-US/gemma-4-12b": {"f1": 70.0} for d in deployment.NER_DATASETS}
    assert deployment.teacher_points(scores, tmp_path) == []
    for dataset in deployment.NER_DATASETS:
        path = timing.timing_path("gemma-4-12b", dataset, "en-US", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seconds_per_sentence": 0.3}))
    [teacher] = deployment.teacher_points(scores, tmp_path)
    assert teacher["latency_ms"] == pytest.approx(300.0)
    assert teacher["f1"] == 70.0


def test_all_points_share_one_panel():
    table = deployment.points(_df(), {}, "no-such-root")
    assert table.timing_protocol.nunique() == 1
    assert table.coverage.nunique() == 1


def test_macros_name_each_point_with_letters_only():
    from active_gliner.analysis import numbers

    macros = deployment.macros(deployment.points(_df(), {}, "no-such-root"))
    assert macros["DeployStudentTrueRandom"] == 81.0
    assert macros["DeployZeroShot"] == 50.0
    assert macros["DeployZeroShotLatency"] == 2.0
    assert "DeploySpeedup" not in macros  # no teacher timing, no Gemma student
    numbers.write_macros("/dev/null", macros)
