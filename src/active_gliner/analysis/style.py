"""One look for every paper figure, modelled on the EmbeddingGemma 2 paper (Figure 2).

+ Serif text that matches the paper body (Times), bold panel titles.
+ A soft slate frame, a light dashed grid, no tick marks.
+ Thick lines, large markers with a white edge, light shading for a gain.
+ Names written next to lines and points; a small framed legend in an empty corner,
  or one legend above a multi-panel figure.

Figures are drawn at the size they print (ACL: one column 3.03 in, full page 6.3 in)
and included without scaling, so a font size here is the size on the page. The colours passed the
colour-blind checks of the dataviz palette validator (ordinal blue ramp; four task hues).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.3  # inches, ACL \textwidth
COLUMN_WIDTH = 3.03  # inches, ACL \columnwidth

INK = "#2b3445"  # text
FRAME = "#9aa3b5"  # axes frame
GRID = "#dde1ea"
LEADER = "#7d869a"  # thin lines from a label to its point

# Share of human labels: light (few) to dark (all). Ordinal ramp, one hue.
SHARE_COLORS = {0: "#86b6ef", 25: "#5598e7", 50: "#2a78d6", 75: "#1c5cab", 100: "#104281"}
TEACHER = "#e34948"  # the LLM teacher, the line to beat
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
TASK_COLORS = {"NER": BLUE, "Relations": ORANGE, "Multi-label": AQUA, "Slots": VIOLET}

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlepad": 5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "text.color": INK,
    "axes.edgecolor": FRAME,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.8,
    "lines.markeredgecolor": "white",
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": FRAME,
    "legend.fancybox": True,
    "legend.borderpad": 0.4,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "patch.linewidth": 0.6,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def apply() -> None:
    plt.rcParams.update(STYLE)


TASK_OF = {
    "cleanconll": "NER",
    "bc5cdr": "NER",
    "mit_movie": "NER",
    "uner_ewt": "NER",
    "crossre": "Relations",
    "hallmarks": "Multi-label",
    "massive": "Slots",
    "boolq": "Decision",
    "clinc150": "Decision",
}


def title(ax, name: str, dataset: str | None = None) -> None:
    """Bold dataset name, then the task type in italics: 'CleanCoNLL (NER)'."""
    task = TASK_OF.get(dataset or "")
    if task is None:
        ax.set_title(name)
        return
    ax.set_title(name + r"  $\mathit{(" + task.replace("-", r"\text{-}") + r")}$")


def panels(count: int, columns: int, width: float, panel_height: float):
    """A figure with `count` panels, `columns` per row; a short last row is centred.

    Returns the figure and the list of axes (no empty panel is left over).
    """
    columns = min(columns, count)
    lines = (count + columns - 1) // columns
    fig = plt.figure(figsize=(width, panel_height * lines), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.04, hspace=0.06)
    # Two grid cells per panel, so a short last row can start half a panel in.
    grid = fig.add_gridspec(lines, 2 * columns)
    axes = []
    for index in range(count):
        line, place = divmod(index, columns)
        in_line = min(columns, count - line * columns)
        start = (columns - in_line) + 2 * place
        axes.append(fig.add_subplot(grid[line, start : start + 2]))
    return fig, axes


def top_legend(fig, handles, labels, columns: int) -> None:
    """One frameless legend for a multi-panel figure, in a row above the panels."""
    fig.legend(handles, labels, loc="outside upper center", ncol=columns, frameon=False)


def label_point(ax, x, y, text, dx=6, dy=6, ha="left") -> None:
    """Name a point, with a thin leader line when the label sits away from it."""
    far = abs(dx) > 10 or abs(dy) > 10
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": LEADER, "linewidth": 0.6} if far else None,
        annotation_clip=False,
    )


def spread(values: list[float], gap: float) -> list[float]:
    """Move label positions apart by at least `gap`, keeping their order and centre."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    shift = (sum(values) - sum(placed)) / len(values) if values else 0.0
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = placed[rank] + shift
    return out


def label_at(ax, point, place, text, ha="left") -> None:
    """Write `text` at `place` (data units), with a thin leader line back to `point`."""
    ax.annotate(
        text,
        point,
        xytext=place,
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=8,
        color=INK,
        arrowprops={
            "arrowstyle": "-",
            "color": LEADER,
            "linewidth": 0.6,
            "shrinkA": 1,
            "shrinkB": 3,
        },
        annotation_clip=False,
    )


def compact(value, _position=None) -> str:
    """100 -> 100, 1000 -> 1k, 10000 -> 10k: short tick labels on a log axis."""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


apply()
