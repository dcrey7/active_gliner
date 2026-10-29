"""The shared figure style: label spreading, tick labels and the panel layout."""

import matplotlib.pyplot as plt

from active_gliner.analysis import style


def test_spread_keeps_order_and_gap():
    values = [80.0, 81.0, 79.5, 40.0]
    placed = style.spread(values, gap=3.0)
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranked = [placed[i] for i in order]
    assert all(b - a >= 3.0 - 1e-9 for a, b in zip(ranked, ranked[1:], strict=False))
    # Labels far from the others do not move apart from the shared shift.
    assert abs(sum(placed) - sum(values)) < 1e-9


def test_compact_tick_labels():
    assert style.compact(100) == "100"
    assert style.compact(1000) == "1k"
    assert style.compact(2500) == "2.5k"
    assert style.compact(10000) == "10k"


def test_panels_leave_no_empty_slot():
    fig, axes = style.panels(7, 4, style.TEXT_WIDTH, panel_height=1.6)
    assert len(axes) == 7
    assert len(fig.axes) == 7
    plt.close(fig)


def test_title_names_the_task():
    fig, ax = plt.subplots()
    style.title(ax, "Hallmarks", "hallmarks")
    assert "Multi" in ax.get_title()
    style.title(ax, "Other", "unknown")
    assert ax.get_title() == "Other"
    plt.close(fig)
