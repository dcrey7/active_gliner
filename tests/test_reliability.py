import pandas as pd

from active_gliner.analysis import reliability


def test_binned_counts_and_share_correct():
    pairs = [(0.95, True), (0.97, False), (0.72, False), (1.0, True)]
    table = reliability.binned(pairs)
    assert list(table["count"]) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 3]
    top = table[table.bin == 9].iloc[0]
    assert abs(top.correct - 2 / 3) < 1e-9


def test_macros_use_lowest_filled_bin_and_top_bin():
    pairs = [(0.75, False)] * 15 + [(0.75, True)] * 5 + [(0.95, True)] * 30 + [(0.35, True)]
    table = reliability.binned(pairs)
    macros = reliability.macros({("mit_movie", "en-US"): {"zero-shot": table}})
    # The 0.3 bin has one item, under the 20-item floor, so Low starts at 0.7.
    assert macros == {
        "MITMovieReliabilityZeroShotLow": 25.0,
        "MITMovieReliabilityZeroShotLowFrom": 0.7,
        "MITMovieReliabilityZeroShotHigh": 100.0,
    }


def test_trained_dirs_pick_main_random_ground_truth_runs():
    df = pd.DataFrame(
        [
            dict(
                dataset="cleanconll",
                locale="en-US",
                labels="ground_truth",
                selector="random",
                n=400,
                gt_fraction=None,
                variant=None,
                run_dir="keep",
            ),
            dict(
                dataset="cleanconll",
                locale="en-US",
                labels="ground_truth",
                selector="random",
                n=100,
                gt_fraction=None,
                variant=None,
                run_dir="other budget",
            ),
            dict(
                dataset="cleanconll",
                locale="en-US",
                labels="ground_truth",
                selector="random",
                n=400,
                gt_fraction=None,
                variant="heads-only",
                run_dir="variant",
            ),
            dict(
                dataset="cleanconll",
                locale="en-US",
                labels="gemma-4-12b",
                selector="random",
                n=400,
                gt_fraction=None,
                variant=None,
                run_dir="teacher labels",
            ),
        ]
    )
    assert reliability.trained_dirs(df, "cleanconll", "en-US") == ["keep"]
