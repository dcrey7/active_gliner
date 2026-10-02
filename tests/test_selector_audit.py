import json

import pandas as pd

from active_gliner.analysis import selector_audit


def _run(tmp_path, name, ids):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "selected_ids.json").write_text(json.dumps(ids))
    return str(run_dir)


def _row(run_dir, selector, task="ner", seed=1):
    return dict(
        dataset="mit_movie",
        locale="en-US",
        labels="ground_truth",
        task=task,
        n=2,
        seed=seed,
        selector=selector,
        gt_fraction=None,
        run_dir=run_dir,
    )


def test_identity_counts_shared_selections(tmp_path):
    same = ["a", "b"]
    rows = [
        _row(_run(tmp_path, "avg", same), "avg"),
        _row(_run(tmp_path, "mse", same), "mse"),
        _row(_run(tmp_path, "mnlp", list(reversed(same))), "mnlp"),
        _row(_run(tmp_path, "min", same), "min"),
        # Seed 2: mnlp picks a different set, so the cell is not identical.
        _row(_run(tmp_path, "avg2", same), "avg", seed=2),
        _row(_run(tmp_path, "mse2", same), "mse", seed=2),
        _row(_run(tmp_path, "mnlp2", ["a", "c"]), "mnlp", seed=2),
    ]
    macros = selector_audit.identity_macros(pd.DataFrame(rows))
    assert macros["ThesisSelectorCells"] == 2
    assert macros["ThesisSelectorIdenticalCells"] == 1
    assert macros["ThesisSelectorNERCells"] == 1
    assert macros["ThesisSelectorNERMinIdenticalCells"] == 1


def test_identity_ignores_min_with_empty_sentences_last(tmp_path):
    rows = [_row(_run(tmp_path, s, ["a"]), s) for s in ("avg", "mse", "mnlp", "min")]
    for row in rows:
        row["no_prediction"] = "zero"
    last = _row(_run(tmp_path, "min-last", ["z"]), "min")
    last["no_prediction"] = "last"
    macros = selector_audit.identity_macros(pd.DataFrame([*rows, last]))
    assert macros["ThesisSelectorNERMinIdenticalCells"] == 1


def test_identity_skips_mixing_runs(tmp_path):
    rows = [_row(_run(tmp_path, s, ["a"]), s) for s in ("avg", "mse", "mnlp")]
    for row in rows:
        row["gt_fraction"] = 0.5
    assert selector_audit.identity_macros(pd.DataFrame(rows)) == {}


def test_spread_is_range_over_thesis_selectors():
    means = {"MITMovieGemmaAvg": 74.1, "MITMovieGemmaMse": 74.4, "MITMovieGemmaMnlp": 74.0}
    spread = selector_audit.spread_macros(means)
    assert abs(spread["MITMovieGemmaThesisSpread"] - 0.4) < 1e-9
    assert "MITMovieGroundTruthThesisSpread" not in spread


def test_no_prediction_share_from_pool_scores(tmp_path):
    pool = tmp_path / "pool.jsonl"
    lines = [{"id": str(i), "has_prediction": i >= 3} for i in range(4)]
    pool.write_text("".join(json.dumps(line) + "\n" for line in lines))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "acquisition.json").write_text(json.dumps({"pool_score_artifact": str(pool)}))
    df = pd.DataFrame([dict(dataset="mit_movie", locale="en-US", run_dir=str(run_dir))])
    macros = selector_audit.no_prediction_macros(df)
    assert macros["MITMovieNoPredictionCount"] == 3
    assert macros["MITMovieNoPredictionShare"] == 75.0
