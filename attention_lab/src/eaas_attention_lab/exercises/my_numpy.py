"""Exercise: NumPy for the attention lab, written and checked by you.

The task sheet is exercises/numpy_for_attention/README.md in the repository folder.
This file holds everything you change: the settings, one function with a mistake to
find, and six functions to write, plus one for a star task. Each run checks them on
the tiny GPT-2 with random weights, against the forward pass in gpt2.py, and prints
one line per function: True, "not written yet", or False with a hint at what to
change.

The file uses only the tiny GPT-2, whatever is in downloads/, and never downloads
anything. tiny_trace() below serves the REPL tasks of the sheet.

Run from the attention_lab folder: uv run my-numpy
"""

import math
from collections.abc import Sequence

import numpy as np

from eaas_attention_lab import checks, show
from eaas_attention_lab.gpt2 import (
    Array,
    Trace,
    forward,
    layer_norm,
    merge_heads,
    softmax,
)

# --- The settings -------------------------------------------------------------------

LAYER: int = 5  # the layer whose matrices the checks use, 0 to 11
HEAD: int = 4  # its head, 0 to 11
WINDOW: int = 2  # task C3: how many tokens before itself a token may read


# --- Given, with a mistake to find (task A3) -----------------------------------------


def scaled_scores(q: Array, k: Array) -> Array:
    """Task A3: S = Q K^T / sqrt(d_h) of one head, [T, T], in float32 like q and k.

    q and k are [T, d_h]. The numbers are right, but the first run says that the
    result is float64. Find out why, then make it float32 with the same numbers.
    """
    d_h = q.shape[-1]
    s: Array = q @ k.T / np.sqrt(d_h)
    return s


# --- Your code ----------------------------------------------------------------------


def embed(weights: dict[str, Array], ids: list[int]) -> Array:
    """Task B2: the input of the first block for the token IDs ids, [T, d].

    Each token's row of the embedding table weights["wte.weight"], [V, d], plus the
    row of the position table weights["wpe.weight"] for its position: row 0 for the
    first token, row 1 for the second, and so on. No loop.
    """
    raise NotImplementedError


def first_token_weights(trace: Trace) -> Array:
    """Task B3: the weight every token gives the first token, in every head.

    trace.attention is [layers, heads, T, T], the weights A of every head of every
    layer. Return [layers, heads, T]: for every layer and head, A[i, 0] of every
    row i. One line, no loop.
    """
    raise NotImplementedError


def layer_norm_rows(x: Array, gain: Array, bias: Array, eps: float) -> Array:
    """Task C2: LayerNorm of every row of x, [T, d], without layer_norm() of gpt2.py.

    Give each row mean 0 and variance 1, with eps added to the variance before the
    square root, then multiply by gain and add bias, both [d]. No loop.
    """
    raise NotImplementedError


def window_mask(t: int, w: int) -> Array:
    """Task C3: an additive mask, [t, t], float32, for sliding-window attention.

    Token i may read itself and the w tokens before it. The mask holds 0 where row
    i may read column j, and -inf everywhere else: in the future, and more than w
    tokens back. No loop.
    """
    raise NotImplementedError


def all_scores(q: Array, k: Array) -> Array:
    """Task D2: the scores S of all heads at once, [heads, T, T], in float32.

    q and k are [heads, T, d_h], one matrix per head. One @ for all heads: no loop,
    and not scaled_scores().
    """
    raise NotImplementedError


def largest_score(s: Array) -> tuple[int, int]:
    """Task D3: the cell (i, j) of the largest score of one head's S, [T, T].

    Row i is the token that reads, column j the token that is read. Return the
    cell as (i, j). No loop.
    """
    raise NotImplementedError


def merge(heads: Array) -> Array:
    """Part D, star task (optional): [heads, T, d_h] to [T, d], no merge_heads().

    Row i of the result holds token i's results of all heads side by side: the d_h
    numbers of head 0 first, then those of head 1, and so on. No loop.
    """
    raise NotImplementedError


# --- The run: nothing to change below this line -------------------------------------

# The checks run on this text, whatever you try in the REPL: 9 tokens with
# <|endoftext|> in front, so that code written for the 6 tokens of "the cat sat on
# it" shows its mistake.
CHECK_TEXT: str = "the dog sat on the mat because it"

FLOAT64 = "a NumPy float64 number in the formula turns float32 arrays into float64"


def tiny_trace(text: str) -> Trace:
    """The forward pass of the tiny GPT-2 for text, with <|endoftext|> in front.

    For the REPL tasks of the sheet: trace = tiny_trace("the cat sat on it"). The
    tiny GPT-2 knows 49 words; any other word stops the run with the list of them.
    """
    model = checks.tiny_model()
    return forward(checks.encode(model, text), model.weights, model.config)


def rows_of(table: Array, numbers: Sequence[int]) -> Array:
    """The rows of table with these numbers, one by one: what a lookup means."""
    return np.array([table[n] for n in numbers])


def window(t: int, w: int, *, future: bool = False) -> Array:
    """A window mask cell by cell: 0 where row i may read column j, else -inf."""
    return np.array(
        [
            [0.0 if (j <= i or future) and i - j <= w else -np.inf for j in range(t)]
            for i in range(t)
        ],
        dtype=np.float32,
    )


def check_parts_a_and_b(
    trace: Trace, weights: dict[str, Array], ids: list[int]
) -> None:
    """Tasks A3, B2 and B3."""
    layer, t = trace.layers[LAYER], len(ids)
    print("\nPart A: one block of numbers")
    checks.check(
        "scaled_scores",
        scaled_scores,
        layer.q[HEAD],
        layer.k[HEAD],
        expected=layer.scores[HEAD],
        dtype=np.float32,
        if_dtype=FLOAT64,
    )

    print("\nPart B: rows are tokens")
    wte, wpe = weights["wte.weight"], weights["wpe.weight"]
    positions = "a token's position is 0 to T-1, whatever its ID"
    checks.check(
        "embed",
        embed,
        weights,
        ids,
        expected=trace.h1,
        mistakes={
            "the first T rows of the embedding table are the first entries of the "
            "vocabulary, not your tokens: pick the rows with the list of token IDs": (
                rows_of(wte, range(t)) + rows_of(wpe, range(t))
            ),
            "add each token's position vector, a row of the position table": rows_of(
                wte, ids
            ),
        },
        if_raised={
            "out of bounds": f"{positions}: take the first T rows of the position "
            "table",
            "could not be broadcast": "the position table has a row for each of the "
            f"{len(wpe)} positions the tiny GPT-2 can read: take only the first T rows",
        },
    )
    layers = trace.layers
    reading = [[[a[i, 0] for i in range(t)] for a in lay.weights] for lay in layers]
    row_0 = [[[a[0, j] for j in range(t)] for a in lay.weights] for lay in layers]
    checks.check(
        "first_token_weights",
        first_token_weights,
        trace,
        expected=np.array(reading),
        mistakes={
            "row 0 is the first token reading; the first token being read is column 0: "
            "A[i, 0] of every row i": np.array(row_0)
        },
        if_shape="one number for every layer, head and reading token: "
        "[layers, heads, T]",
    )


def check_part_c(trace: Trace, weights: dict[str, Array], labels: list[str]) -> None:
    """Tasks C2 and C3; the window printout once window_mask() works."""
    layer = trace.layers[LAYER]
    t = len(labels)
    print("\nPart C: one line for all rows")
    gain, bias = weights[f"h.{LAYER}.ln_1.weight"], weights[f"h.{LAYER}.ln_1.bias"]
    x = layer.h_in.copy()
    x[2] = 0.5  # one row whose numbers are all equal: its variance is 0

    def diagnose(mine: Array) -> str:
        others = [row for i, row in enumerate(mine) if i != 2]
        if all(abs(np.mean(r)) < 1e-3 and abs(np.var(r) - 1) < 1e-2 for r in others):
            return "the rows have mean 0 and variance 1: multiply by gain, add bias"
        undone = (mine - bias) / gain
        if all(abs(np.mean(column)) < 1e-3 for column in undone.T):
            return (
                "each column has mean 0: normalize each token's row instead, along the "
                "last axis (axis=-1)"
            )
        return ""

    whole = gain * (x - x.mean()) / np.sqrt(x.var() + 1e-5) + bias
    checks.check(
        "layer_norm_rows",
        layer_norm_rows,
        x,
        gain,
        bias,
        1e-5,
        expected=layer_norm(x, gain, bias, 1e-5),
        mistakes={
            "one mean and one variance for the whole matrix: take them for each "
            "token's row, along the last axis (axis=-1)": whole
        },
        diagnose=diagnose,
        if_nan="Row 2 has the same number everywhere, so its variance is 0: add eps to "
        "the variance before the square root",
        if_raised={
            "could not be broadcast": "keep the reduced axis with keepdims=True, in "
            "the mean and in the variance, so that a row's value, (T, 1), fits its "
            "row, (T, d)"
        },
    )

    def as_scores(mine: Array) -> str:
        if set(np.unique(mine).tolist()) <= {0.0, 1.0}:
            return (
                "the mask is added to the scores: 0 where a token may read and -inf "
                "where it may not, not 0 and 1"
            )
        return ""

    read = "token i reads itself and the w tokens before it"
    no_self = window(t, WINDOW)
    np.fill_diagonal(no_self, -np.inf)
    works = checks.check(
        "window_mask",
        window_mask,
        t,
        WINDOW,
        expected=window(t, WINDOW),
        mistakes={
            "a token may not read later tokens: the cells above the diagonal need "
            "-inf too": window(t, WINDOW, future=True),
            f"one token too many: {read}": window(t, WINDOW + 1),
            f"one token too few: {read}": window(t, WINDOW - 1),
            "a token may read itself: the future starts above the diagonal": no_self,
        },
        diagnose=as_scores,
        dtype=np.float32,
        if_dtype="the scores are float32, and a float64 mask would turn them into "
        "float64: make the mask float32",
    )
    if works:
        a = softmax(layer.scores[HEAD] + window_mask(t, WINDOW))
        print(
            f"\n  A of head {HEAD} with window_mask({t}, {WINDOW}): each token reads "
            f"itself and the {WINDOW} tokens before it"
        )
        show.print_matrix(a, labels)


def check_part_d(trace: Trace, labels: list[str], d: int) -> None:
    """Tasks D2, D3 and the star task merge; the largest score once D3 works."""
    layer = trace.layers[LAYER]
    t, d_h = len(labels), layer.q.shape[-1]
    print("\nPart D: heads as a stack")
    checks.check(
        "all_scores",
        all_scores,
        layer.q,
        layer.k,
        expected=layer.scores,
        mistakes={
            "divide by the square root of d_h, the width of one head": (
                layer.scores * math.sqrt(d_h)
            ),
            "divide by the square root of d_h, the last axis of q, not of d": (
                layer.scores * math.sqrt(d_h) / math.sqrt(d)
            ),
        },
        dtype=np.float32,
        if_dtype=FLOAT64,
        if_raised={
            "core dimension": ".T reverses all three axes of a stack: swap only the "
            "last two, with transpose(0, 2, 1)"
        },
    )
    s = layer.scores[HEAD]
    _, row, column = max((float(s[i, j]), i, j) for i in range(t) for j in range(t))
    found = checks.check(
        "largest_score",
        largest_score,
        s,
        expected=np.array([row, column]),
        mistakes={
            "the cell is (row, column): the row is the token that reads, the column "
            "the token that is read": np.array([column, row])
        },
        if_shape="return the cell as (row, column): argmax() without an axis gives "
        "one flat index, and np.unravel_index turns it into (row, column)",
    )
    if found:
        i, j = largest_score(s)
        if j > i:
            where = "a later token: the mask hides it"
        else:
            where = "itself" if j == i else "an earlier token: it may read it"
        print(
            f"\n  The largest score of head {HEAD}, layer {LAYER}: S[{i}, {j}] = "
            f"{s[i, j]:.2f}. Token {i}, {labels[i]}, reads token {j}, {labels[j]}, "
            f"{where}"
        )
    print("\nPart D, star task (optional)")
    checks.check(
        "merge",
        merge,
        layer.head_outputs,
        expected=merge_heads(layer.head_outputs),
        mistakes={
            "reshape reads the stack head by head: transpose first, so that each "
            "token's row holds all heads side by side": layer.head_outputs.reshape(t, d)
        },
    )


def main() -> None:
    """Check the functions of this file on the tiny GPT-2."""
    show.utf8_output()
    model = checks.tiny_model()
    ids = checks.encode(model, CHECK_TEXT)
    trace = forward(ids, model.weights, model.config)
    print(
        f"The tiny GPT-2 with random weights, {CHECK_TEXT!r} with <|endoftext|> in "
        f"front: {len(ids)} tokens. Layer {LAYER}, head {HEAD}."
    )
    check_parts_a_and_b(trace, model.weights, ids)
    check_part_c(trace, model.weights, model.entries(ids))
    check_part_d(trace, model.entries(ids), model.config.d)


if __name__ == "__main__":
    main()
