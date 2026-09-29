"""The main results table and cell macros come from runs, never typed by hand (rule 10)."""

import pandas as pd
import pytest

from active_gliner.analysis import numbers, tables


def _row(labels, selector, n, seed, f1, dataset="cleanconll", **kw):
    return {
        "dataset": dataset,
        "locale": "en-US",
        "labels": labels,
        "selector": selector,
        "n": n,
        "seed": seed,
        "f1": f1,
        "gt_fraction": None,
        "no_prediction": "zero",
        **kw,
    }


def _df():
    return pd.DataFrame(
        [
            _row("ground_truth", "random", 400, 1, 80.0),
            _row("ground_truth", "random", 400, 2, 82.0),
            _row("ground_truth", "min", 400, 1, 70.0, no_prediction="last"),
            _row("gemma-4-12b", "min", 400, 1, 60.0),
            _row("gemma-4-12b", "random", 100, 1, 10.0),  # not the main budget
            _row("gemma-4-12b", "min", 400, 1, 99.0, gt_fraction=0.25),  # mixing run
            _row("ground_truth", "all", 13957, 1, 90.0),
            _row("ground_truth", "min", 200, 1, 30.0, dataset="crossre"),
            _row("none", "zero_shot", None, None, 55.0, zero_shot_name="gliner2.5-multi-v1"),
        ]
    )


def test_summary_keeps_main_budget_and_separates_empty_last():
    s = tables.summary(_df()).set_index(["dataset", "labels", "arm"])
    assert s.loc[("cleanconll", "ground_truth", "random")]["mean"] == 81.0
    assert s.loc[("cleanconll", "ground_truth", "random")]["seeds"] == 2
    assert s.loc[("cleanconll", "ground_truth", "min_last")]["mean"] == 70.0
    assert s.loc[("cleanconll", "gemma-4-12b", "min")]["mean"] == 60.0  # mixing run excluded
    assert ("cleanconll", "gemma-4-12b", "random") not in s.index  # N 100 only
    assert s.loc[("crossre", "ground_truth", "min")]["mean"] == 30.0  # CrossRE budget is 200


def test_cell_macros_are_letter_only_and_include_bounds(tmp_path):
    macros = tables.cell_macros(_df())
    assert macros["CleanCoNLLGroundTruthRandom"] == 81.0
    assert macros["CleanCoNLLGroundTruthWholePool"] == 90.0
    assert macros["CleanCoNLLZeroShot"] == 55.0
    numbers.write_macros(tmp_path / "n.tex", macros)  # raises on a bad name


def test_bootstrap_p_of_zero_is_a_bound():
    from active_gliner.analysis.report import format_p

    assert format_p(0.0, 1000) == "$<0.001$"
    assert format_p(0.336, 1000) == "$=0.336$"


def test_e4b_students_get_macros_but_no_table_row():
    df = pd.DataFrame(_df().to_dict("records") + [_row("gemma-4-e4b", "random", 400, 1, 68.0)])
    assert tables.cell_macros(df)["CleanCoNLLEfourBRandom"] == 68.0
    assert "E4B" not in tables.main_table(df)


def test_mixing_macros_split_by_fraction_and_assignment():
    df = pd.DataFrame(
        [
            _row("gemma-4-12b", "min", 400, 1, 70.0, gt_fraction=0.25, gt_assignment="random"),
            _row("gemma-4-12b", "min", 400, 2, 72.0, gt_fraction=0.25, gt_assignment="random"),
            _row("gemma-4-12b", "min", 400, 1, 60.0, gt_fraction=0.25, gt_assignment="routed"),
            _row("gemma-4-12b", "min", 400, 1, 80.0, gt_fraction=0.5, gt_assignment="random"),
        ]
    )
    macros = tables.mixing_macros(df)
    assert macros == {
        "CleanCoNLLMixTwentyFive": 71.0,
        "CleanCoNLLMixTwentyFiveRouted": 60.0,
        "CleanCoNLLMixFifty": 80.0,
    }


def test_threshold_spread_macros():
    df = pd.DataFrame(
        [
            _row("ground_truth", "random", 400, 1, 80.0, dev_threshold_spread=0.2),
            _row("ground_truth", "random", 400, 2, 81.0, dev_threshold_spread=0.4),
            _row(
                "none",
                "zero_shot",
                None,
                None,
                55.0,
                zero_shot_name="gliner2.5-multi-v1",
                dev_threshold_spread=9.0,
            ),
        ]
    )
    macros = tables.threshold_macros(df)
    assert macros["CleanCoNLLZeroShotSpread"] == 9.0
    assert macros["CleanCoNLLTrainedSpread"] == pytest.approx(0.3)


def test_selector_and_variant_macros():
    df = pd.DataFrame(
        [
            _row("ground_truth", "mnlp", 400, 1, 70.0),
            _row("ground_truth", "mnlp", 400, 2, 74.0),
            _row("ground_truth", "mnlp", 100, 1, 10.0),  # not the main budget
            _row("gemma-4-12b", "mse", 400, 1, 60.0),
        ]
    )
    assert tables.selector_macros(df) == {
        "CleanCoNLLGroundTruthMnlp": 72.0,
        "CleanCoNLLGemmaMse": 60.0,
    }
    variants = pd.DataFrame(
        [
            _row("ground_truth", "random", 400, 1, 80.0, variant="heads-only"),
            _row("ground_truth", "all", 13957, 1, 90.0, variant="full-finetune"),
        ]
    )
    assert tables.variant_macros(variants) == {
        "CleanCoNLLHeadsOnly": 80.0,
        "CleanCoNLLFullFinetune": 90.0,
    }


def test_teacher_and_ladder_macros():
    scores = {"bc5cdr/en-US/gemma-4-12b": {"f1": 79.0}}
    bins = {"bc5cdr/en-US/gemma-4-12b": [{"bin": 1, "f1": 0.9}, {"bin": 0, "f1": 0.7}]}
    macros = tables.teacher_macros(scores, bins)
    assert macros == {
        "TeacherBCfiveCDR": 79.0,
        "TeacherBCfiveCDRLeastSure": 70.0,
        "TeacherBCfiveCDRMostSure": 90.0,
    }
    entry = {
        "n": 764,
        "f1": {"gemma-4-12b": 82.0, "gemma-4-e4b": 76.0},
        "error_overlap": {"gemma-4-12b": {"gemma-4-e4b": 0.6}},
        "gap": {"mean": 6.0, "interval": {"low": 3.0, "high": 9.0}},
    }
    macros = tables.ladder_macros({"bc5cdr": entry})
    assert macros["LadderBCfiveCDRTwelveB"] == 82.0
    assert macros["LadderBCfiveCDRGapLow"] == 3.0
    assert macros["LadderBCfiveCDROverlap"] == 0.6
    numbers.write_macros("/dev/null", macros)  # raises on a bad name


def test_main_table_shows_mean_sd_and_seeds():
    text = tables.main_table(_df())
    assert "81.0 $\\pm$ 1.4 (2)" in text
    assert "CleanCoNLL & Ground truth & 55.0" in text
    assert text.count("\\\\") == 1 + 2 * len(tables.DATASETS)
