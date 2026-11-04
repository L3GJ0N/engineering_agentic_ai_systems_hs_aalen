"""Show results: matrices as tables in the terminal, figures as PNG files.

All figures follow the same rules, so that two figures can be compared:
  - In a matrix of scores or weights, row i belongs to the token that reads
    (its query), column j to the token that is read (its key).
  - Weights use one colour, from white for 0 to dark blue for 1, always on
    this scale: the same colour means the same weight in every figure.
  - Values with a sign, such as scores, are red below 0, grey at 0 and blue
    above 0, on a scale symmetric around 0.
  - Cells the causal mask hides are grey, a darker grey than 0.

Figures are written to attention_lab/outputs/ and never shown in a window,
which works on every system, with or without a screen. Open them in VS Code
or any image viewer. Each run overwrites its figures: to compare two
settings, copy a figure first.
"""

import io
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from eaas_attention_lab.gpt2 import Array

OUTPUTS: Path = Path(__file__).resolve().parents[2] / "outputs"

# --- One look for every figure -------------------------------------------------

SURFACE = "#fcfcfb"  # background
INK = "#0b0b0b"  # titles and values
INK_SECONDARY = "#52514e"  # labels
INK_MUTED = "#898781"  # ticks and notes
HIDDEN = "#c3c2b7"  # cells the causal mask hides
GRID = "#e1e0d9"  # gridlines
# Colours for up to three data series, always in this order. They stay
# distinguishable for the common kinds of colour blindness.
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

WEIGHTS_COLOURS = LinearSegmentedColormap.from_list(
    "weights",
    [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"],
).with_extremes(bad=HIDDEN)
SIGNED_COLOURS = LinearSegmentedColormap.from_list(
    "signed",
    ["#8c2323", "#e34948", "#f2a5a0", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"],
).with_extremes(bad=HIDDEN)

rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": HIDDEN,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def utf8_output() -> None:
    """Let the Windows terminal print vocabulary entries such as Ġcat."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


def future(t: int) -> NDArray[np.bool_]:
    """[T, T]: True above the diagonal, where token j comes after token i."""
    return np.triu(np.ones((t, t), dtype=np.bool_), k=1)


# --- In the terminal ---------------------------------------------------------------


def print_matrix(
    matrix: Array, labels: Sequence[str], hide_future: bool = False
) -> None:
    """Print a matrix with one numbered row per token, labelled with the token.

    In a [T, T] matrix of scores or weights, column j belongs to the token of
    row j. In a [T, n] matrix, such as some columns of Q, the columns are
    numbered 0 to n - 1.
    """
    t, n = matrix.shape
    width = max(len(label) for label in labels)
    print(" " * (width + 4) + "".join(f"{j:>7}" for j in range(n)))
    hidden = future(t)
    for i, label in enumerate(labels):
        cells = (
            "      ·" if hide_future and hidden[i, j] else f"{matrix[i, j]:7.2f}"
            for j in range(n)
        )
        print(f"{i:>2}  {label:<{width}}" + "".join(cells))
    if hide_future:
        print(" " * (width + 4) + "· hidden by the mask: these tokens come later")


# --- In figures ----------------------------------------------------------------------


def figure(width: float, height: float) -> Figure:
    """A new figure, in inches. It belongs to no window and needs no closing."""
    return Figure(figsize=(width, height), layout="constrained")


def text_colour(background: tuple[float, float, float]) -> str:
    """Dark or white text, whichever contrasts more with the background."""

    def linear(channel: float) -> float:  # undo the sRGB gamma curve
        if channel <= 0.04045:
            return channel / 12.92
        return float(((channel + 0.055) / 1.055) ** 2.4)

    r, g, b = (linear(c) for c in background)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Contrast ratios as in the web's accessibility guidelines (WCAG).
    with_white = 1.05 / (luminance + 0.05)
    with_ink = (luminance + 0.05) / 0.0533
    return "white" if with_white > with_ink else INK


NUMBERED_TOKENS = 8  # heatmaps of more tokens get no numbers in their cells


def heatmap(
    ax: Axes,
    matrix: Array,
    labels: Sequence[str],
    *,
    signed: bool = False,
    hide_future: bool = False,
    numbers: bool = True,
) -> AxesImage:
    """Draw a [T, T] matrix of the tokens in labels as coloured cells.

    signed=False: weights, on the fixed scale 0 to 1. signed=True: red, grey
    and blue, symmetric around 0, up to the largest absolute value.
    hide_future greys out the cells above the diagonal; numbers writes each
    visible value into its cell, for up to NUMBERED_TOKENS tokens: beyond
    that, the numbers would overlap.
    """
    t = len(labels)
    hidden = future(t) if hide_future else np.zeros((t, t), dtype=np.bool_)
    values = np.ma.masked_array(matrix, mask=hidden)
    if signed:
        top = float(np.abs(values).max()) or 1.0
        norm, colours = Normalize(-top, top), SIGNED_COLOURS
    else:
        norm, colours = Normalize(0.0, 1.0), WEIGHTS_COLOURS
    image = ax.imshow(values, cmap=colours, norm=norm)

    ax.set_xticks(range(t), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(t), labels)
    ax.set_xlabel("key: the token that is read")
    ax.set_ylabel("query: the token that reads")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if numbers and t <= NUMBERED_TOKENS:
        for i, j in np.ndindex(t, t):
            if not hidden[i, j]:
                background = to_rgb(colours(norm(float(matrix[i, j]))))
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_colour(background),
                )
    return image


def outline_future(ax: Axes, t: int) -> None:
    """Draw a staircase line between the visible cells and the future ones."""
    xs: list[float] = []
    ys: list[float] = []
    for i in range(t - 1):  # row i: the future starts right of column i
        xs += [i + 0.5, i + 0.5]
        ys += [i - 0.5, i + 0.5]
    xs.append(t - 0.5)
    ys.append(t - 1.5)
    ax.plot(xs, ys, color=INK, linewidth=1.2)


def colour_bar(
    fig: Figure, image: AxesImage, axes: Axes | Sequence[Axes], label: str
) -> None:
    """Add the colour scale of image beside axes."""
    bar = fig.colorbar(image, ax=axes, shrink=0.8)
    bar.set_label(label, color=INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)


# --- The figures the commands draw -----------------------------------------------


def row_notes(ax: Axes, notes: Sequence[str]) -> None:
    """Write a note beside each row of a heatmap, on its right-hand side."""
    side = ax.secondary_yaxis("right")
    side.set_yticks(range(len(notes)), list(notes))
    side.tick_params(length=0)
    side.spines["right"].set_visible(False)


def scores_and_weights(
    scores: Array, weights: Array, late: Array, labels: Sequence[str], title: str
) -> Figure:
    """One head, three panels: S, A, and A with the mask after the softmax.

    The scores show every cell, with a line where the future begins. The
    third panel writes each row's sum beside the row.
    """
    t = len(labels)
    fig = figure(15, 5.2)
    left, middle, right = fig.subplots(1, 3)
    image = heatmap(left, scores, labels, signed=True)
    outline_future(left, t)
    left.set_title("S: scores. Right of the line: the future")
    colour_bar(fig, image, left, "score")
    image = heatmap(middle, weights, labels, hide_future=True)
    middle.set_title("A = softmax(S + mask): rows sum to 1")
    heatmap(right, late, labels, hide_future=True)
    row_notes(right, [f"sum {value:.2f}" for value in late.sum(axis=-1)])
    right.set_title("Mask after the softmax: earlier rows lose weight")
    for ax in (middle, right):
        ax.set_ylabel("")  # the same as on the left
    colour_bar(fig, image, [middle, right], "attention weight")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def head_grid(weights: Array, labels: Sequence[str], title: str) -> Figure:
    """The weights A of every head of one layer, [heads, T, T], four per row."""
    heads = len(weights)
    rows = (heads + 3) // 4
    fig = figure(13, 10 * rows / 3)
    axes = list(fig.subplots(rows, 4, squeeze=False).flat)
    images = [
        heatmap(ax, weights[head], labels, hide_future=True, numbers=False)
        for head, ax in enumerate(axes[:heads])
    ]
    for head, ax in enumerate(axes[:heads]):
        ax.set_title(f"head {head}", fontsize=10)
        if head % 4:
            ax.set_ylabel("")  # only the first column names the axes
        if head < heads - 4:
            ax.set_xlabel("")  # only the last row does
    for ax in axes[heads:]:
        ax.set_visible(False)
    colour_bar(fig, images[0], axes, "attention weight")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def layer_bars(series: Mapping[str, Array], title: str, ylabel: str) -> Figure:
    """One group of bars per layer, one bar per series, for values from 0 to 1.

    Each series holds one value per layer, such as a share averaged over the
    layer's heads. The series get the colours of SERIES, in order.
    """
    fig = figure(9, 3.8)
    ax = fig.subplots()
    width = 0.76 / len(series)  # of one bar, so that a group fills 0.76
    for n, (label, values) in enumerate(series.items()):
        layers = np.arange(len(values))
        offset = (n - (len(series) - 1) / 2) * width
        ax.bar(layers + offset, values, width - 0.04, color=SERIES[n], label=label)
        ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title(title)
    return fig


def labelled_bars(
    values: Mapping[str, float],
    notes: Sequence[str],
    title: str,
    xlabel: str,
    unit: tuple[float, str] = (1.0, ""),
) -> Figure:
    """One horizontal bar per entry, from the top in the given order.

    notes holds the text written at the end of each bar. unit scales the
    numbers on the axis: (1e6, " M") writes 10,000,000 as 10 M.
    """
    fig = figure(8.5, 0.2 + 0.6 * len(values))
    ax = fig.subplots()
    names, sizes = list(values)[::-1], list(values.values())[::-1]
    ax.barh(names, sizes, color=SERIES[0], height=0.6)
    for y, (size, note) in enumerate(zip(sizes, list(notes)[::-1], strict=True)):
        ax.annotate(
            note,
            (size, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    scale, name = unit
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value / scale:g}{name}")
    ax.margins(x=0.22)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    return fig


def save(fig: Figure, name: str) -> Path:
    """Write the figure to outputs/<name>.png and print where it is."""
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"Figure: {path}")
    return path
