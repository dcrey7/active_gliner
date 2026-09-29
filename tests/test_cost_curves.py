"""Labelling cost for the F1-versus-cost figures (design 3f and the release chart).

Human labels use the stated per-sentence rate. Local teachers use free-GPU timing
files, never cached request times, which were measured on a shared GPU.
"""

import json

import httpx
import pandas as pd
import pytest

from active_gliner import data, timing
from active_gliner.analysis import cost, figures


def _write_split(root, dataset="toy", locale="en-US", pool=1000):
    path = root / "splits" / f"{dataset}-{locale}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pool": list(range(pool)), "dev": [], "test": []}))


def _write_timing(root, teacher="gemma-4-12b", dataset="toy", seconds=0.2):
    path = timing.timing_path(teacher, dataset, "en-US", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seconds_per_sentence": seconds}))


def _row(**kw):
    return {
        "labels": "ground_truth",
        "dataset": "toy",
        "locale": "en-US",
        "n": 400,
        "selector": "random",
        **kw,
    }


def test_ground_truth_uses_the_stated_human_rate():
    rate = cost.prices()["human"]["usd_per_sentence"]
    assert rate > 0
    assert cost.annotation_usd(_row(n=400)) == pytest.approx(400 * rate)


def test_local_teacher_needs_a_free_gpu_timing(tmp_path):
    row = _row(labels="gemma-4-12b")
    assert cost.annotation_usd(row, tmp_path) is None
    _write_timing(tmp_path, seconds=0.2)
    gpu = cost.prices()["gpu"]["usd_per_hour"]
    assert cost.annotation_usd(row, tmp_path) == pytest.approx(400 * 0.2 / 3600 * gpu)


def test_api_teacher_has_no_timing_based_cost(tmp_path):
    assert cost.annotation_usd(_row(labels="qwen-3.8-27b"), tmp_path) is None


def test_min_pays_for_the_pool_scoring_pass(tmp_path):
    _write_split(tmp_path, pool=1000)
    df = pd.DataFrame(
        [
            {
                "zero_shot_name": cost.STUDENT_ZERO_SHOT,
                "dataset": "toy",
                "locale": "en-US",
                "latency_ms": 10.0,
            }
        ]
    )
    gpu = cost.prices()["gpu"]["usd_per_hour"]
    assert cost.selection_usd(_row(selector="random"), df, tmp_path) == 0.0
    assert cost.selection_usd(_row(selector="min"), df, tmp_path) == pytest.approx(
        1000 * 0.010 / 3600 * gpu
    )


def test_missing_scoring_time_leaves_cost_missing_not_zero(tmp_path):
    _write_split(tmp_path)
    df = pd.DataFrame([_row(selector="min", zero_shot_name=None, latency_ms=None)])
    assert cost.labelling_cost(df, tmp_path).isna().all()


def test_cost_curves_draw_from_the_cost_column(tmp_path):
    rows = [
        _row(
            labels=label,
            selector=selector,
            n=n,
            seed=seed,
            f1=50.0 + seed,
            gt_fraction=None,
            no_prediction="zero",
            labelling_cost_usd=n * price,
        )
        for label, price in (("ground_truth", 0.5), ("gemma-4-12b", 0.001))
        for selector in ("random", "min")
        for n in (100, 400)
        for seed in (1, 2)
    ]
    paths = figures.cost_curves(pd.DataFrame(rows), "toy", tmp_path)
    assert [p.name for p in paths] == ["cost-toy-en-US.pdf"]
    assert paths[0].exists()


def test_learning_curves_draw_whole_pool_and_zero_shot_lines(tmp_path, monkeypatch):
    lines = []
    real = figures.plt.Axes.axhline

    def spy(self, y, **kw):
        lines.append((kw.get("label"), y))
        return real(self, y, **kw)

    monkeypatch.setattr(figures.plt.Axes, "axhline", spy)
    rows = [
        _row(selector="random", n=100, seed=1, f1=70.0, gt_fraction=None, no_prediction="zero"),
        _row(selector="all", n=1000, seed=1, f1=90.0, gt_fraction=None, no_prediction="zero"),
        _row(
            labels="none",
            selector="zero_shot",
            n=0,
            seed=None,
            f1=55.0,
            zero_shot_name=cost.STUDENT_ZERO_SHOT,
            gt_fraction=None,
            no_prediction="zero",
        ),
    ]
    figures.learning_curves(pd.DataFrame(rows), "toy", tmp_path)
    assert sorted(lines) == [("whole pool, ground_truth", 90.0), ("zero-shot student", 55.0)]


def test_time_teacher_labels_with_an_empty_cache(tmp_path):
    if not data.raw_available("mit_movie"):
        pytest.skip("needs MIT Movie data")

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "fake"}]})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"entities": []})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
        )

    path = timing.time_teacher(
        "gemma-4-12b", "mit_movie", out_root=tmp_path, transport=httpx.MockTransport(handler)
    )
    result = json.loads(path.read_text())
    assert result["sentences"] == timing.SENTENCES
    assert result["seconds_per_sentence"] > 0
    assert result["prompt_tokens"] == 100 * timing.SENTENCES
    assert (
        timing.seconds_per_sentence("gemma-4-12b", "mit_movie", "en-US", tmp_path)
        == (result["seconds_per_sentence"])
    )
