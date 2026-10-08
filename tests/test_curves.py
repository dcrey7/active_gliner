import pandas as pd

from active_gliner.analysis import curves


def _row(labels, selector, n, f1, seed=1, **extra):
    row = dict(
        dataset="cleanconll",
        locale="en-US",
        labels=labels,
        selector=selector,
        n=n,
        seed=seed,
        f1=f1,
        gt_fraction=None,
        gt_assignment=None,
        no_prediction="zero",
        variant=None,
    )
    row.update(extra)
    return row


def test_curve_rows_keep_ranked_random_and_long_whole_pool():
    df = pd.DataFrame(
        [
            _row("ground_truth", "min", 100, 81.0, no_prediction="last"),
            _row(
                "gemma-4-12b",
                "min",
                100,
                75.0,
                gt_fraction=0.25,
                gt_assignment="random",
                no_prediction="last",
            ),
            _row("gemma-4-12b", "random", 100, 69.0),
            _row("ground_truth", "all", 1, 90.0, variant="long"),
            # Left out: min with empty sentences first, routed mixing, short whole pool,
            # thesis variants and other teachers.
            _row("ground_truth", "min", 100, 60.0),
            _row(
                "gemma-4-12b",
                "min",
                400,
                59.0,
                gt_fraction=0.25,
                gt_assignment="routed",
                no_prediction="last",
            ),
            _row("ground_truth", "all", 1, 89.0),
            _row("ground_truth", "random", 400, 70.0, variant="heads-only"),
            _row("gemma-4-e4b", "random", 400, 68.0),
        ]
    )
    rows = curves.curve_rows(df)
    got = sorted(zip(rows.rule, rows.share, rows.budget.astype(str), rows.f1, strict=True))
    assert got == [
        ("random", 0, "100", 69.0),
        ("ranked", 25, "100", 75.0),
        ("ranked", 100, "100", 81.0),
        ("ranked", 100, "all", 90.0),
    ]


def test_curve_macros_name_share_and_budget():
    df = pd.DataFrame(
        [
            _row("ground_truth", "min", 1000, 86.0, seed=1, no_prediction="last"),
            _row("ground_truth", "min", 1000, 88.0, seed=2, no_prediction="last"),
            _row(
                "gemma-4-12b",
                "all",
                1,
                79.0,
                gt_fraction=0.25,
                gt_assignment="random",
                variant="long",
            ),
            _row("gemma-4-12b", "random", 100, 69.0),
        ]
    )
    macros = curves.macros(curves.summary(curves.curve_rows(df)))
    assert macros == {
        "CleanCoNLLCurveHundredAtThousand": 87.0,
        "CleanCoNLLCurveTwentyFiveAtWholePool": 79.0,
    }


def test_figure_writes_a_panel_per_dataset(tmp_path):
    df = pd.DataFrame(
        [
            _row("ground_truth", "min", 100, 81.0, no_prediction="last"),
            _row("ground_truth", "min", 1000, 86.0, no_prediction="last"),
            _row("gemma-4-12b", "random", 100, 69.0),
            _row("ground_truth", "all", 1, 90.0, variant="long"),
        ]
    )
    path = curves.figure(
        curves.curve_rows(df),
        {("cleanconll", "en-US"): 76.1},
        {("cleanconll", "en-US"): 13957},
        tmp_path,
    )
    assert path is not None and path.exists()


def _table():
    df = pd.DataFrame(
        [
            _row("gemma-4-12b", "min", 100, 70.0, no_prediction="last"),
            _row(
                "gemma-4-12b",
                "min",
                100,
                74.0,
                gt_fraction=0.5,
                gt_assignment="random",
                no_prediction="last",
            ),
            _row("ground_truth", "min", 100, 80.0, no_prediction="last"),
            _row(
                "gemma-4-12b",
                "min",
                400,
                78.0,
                gt_fraction=0.25,
                gt_assignment="random",
                no_prediction="last",
            ),
            _row("gemma-4-12b", "all", 1, 74.0, variant="long"),
            _row("ground_truth", "all", 1, 90.0, variant="long"),
            _row("ground_truth", "random", 100, 79.0),
        ]
    )
    return curves.curve_rows(df)


def test_ranked_cells_count_human_and_total_sentences():
    cells = curves.ranked_cells(_table())
    got = sorted(zip(cells.human, cells.total, strict=True))
    assert got == [(0, 100), (50, 100), (100, 100), (100, 400)]


def test_smallest_share_reads_the_grid_against_the_teacher():
    table = curves.summary(_table())
    shares = curves.smallest_share(table, {("cleanconll", "en-US"): 76.0})
    assert shares == {("cleanconll", "en-US", 100): 100, ("cleanconll", "en-US", 400): 25}
    assert curves.random_macros(table) == {"CleanCoNLLCurveHundredAtHundredRandom": 79.0}


def test_latex_table_shows_references_shares_and_dev_choice(tmp_path):
    table = curves.summary(_table())
    chosen = {
        ("cleanconll", "en-US"): dict(
            human=100, total=400, share=0.25, f1=78.0, test=dict(gap=1.0, low=0.5, high=1.5)
        )
    }
    path = curves.latex_table(
        table, {("cleanconll", "en-US"): 76.0}, chosen, tmp_path / "curves.tex"
    )
    row = next(line for line in path.read_text().splitlines() if line.startswith("CleanCoNLL"))
    assert row == (
        "CleanCoNLL & 76.0 & 74.0 & 90.0 & 100\\% & 25\\% & -- & -- & 100+300 & 78.0 \\\\"
    )
    assert curves.fewest_macros(chosen)["CleanCoNLLFewestTeacher"] == 300
