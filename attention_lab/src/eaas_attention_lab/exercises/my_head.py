"""Exercise: one attention head, explored and then written by you.

The task sheet is exercises/one_head/README.md in the repository folder. This
file holds everything you change: the settings, the head as one_head.py
computes it, and three functions for you to write. Each run prints the head's
S, A and result for TEXT, says which rows of A change for a second text,
COMPARE, and checks your functions against the forward pass in gpt2.py.

If GPT-2 small is downloaded, the run uses it. If not, it uses a tiny GPT-2
with random weights, which needs no download. This file never downloads
anything: uv run model-weights does that, once.

Run from the attention_lab folder: uv run my-head
"""

import math

import numpy as np

from eaas_attention_lab import checks, show
from eaas_attention_lab.gpt2 import Array, Trace, causal_mask, forward, softmax
from eaas_attention_lab.model_files import Model

# --- The settings -------------------------------------------------------------------

TEXT: str = "the cat sat on it"
# Task 2: a second text. The run says which rows of A differ from those of TEXT;
# it starts as TEXT itself.
COMPARE: str = "the cat sat on it"
LAYER: int = 5  # 0 to 11, counted from 0, as in Python
HEAD: int = 4  # 0 to 11
# Task 6: the position of the token, counted from 0, that the last token reads.
FIND: int = 2
# Task 6: True runs the tiny GPT-2 even if GPT-2 small is downloaded.
USE_TINY: bool = False


# --- The head, as steps 3, 4 and 7 of one_head.py compute it ----------------------
# Task 5 changes these lines on purpose. The checks of your code do not use them.


def head(q: Array, k: Array, v: Array) -> tuple[Array, Array, Array]:
    """S, A and the result A V of one head, from its queries, keys and values.

    q, k and v are [T, d_h]: one row per token, d_h numbers each. Returns S
    and A, [T, T], and the result, [T, d_h].
    """
    t, d_h = q.shape
    s = q @ k.T / math.sqrt(d_h)  # the scores: every query against every key
    a = softmax(s + causal_mask(t))  # the weights: hide the future, then softmax
    out = a @ v  # each token's result: its row of A times the value rows
    return s, a, out


# --- Your code ----------------------------------------------------------------------


def newest_row(q_new: Array, k: Array, v: Array) -> Array:
    """Task 3: what the head gives the newest token, from its query alone.

    q_new is the newest token's query, [d_h]. k and v hold the keys and the
    values of all tokens so far, the newest one included, [T, d_h] each.
    Return the newest token's result, [d_h]: the last row of A V. Use NumPy,
    not softmax() or head().
    """
    raise NotImplementedError


def attention_weights(s: Array) -> Array:
    """Task 4: the weights A from the scores S, [T, T], for all tokens at once.

    Hide the future, then turn each row into weights that are at least 0 and
    sum to 1. Use NumPy, not softmax() or causal_mask() from gpt2.py.
    """
    raise NotImplementedError


def find_head(trace: Trace, reader: int, read: int) -> tuple[int, int]:
    """Task 6: the layer and head in which token reader gives token read most weight.

    trace.layers[n].weights[h] is A of head h in layer n, [T, T]. Return
    (n, h) with the largest A[reader, read] of all layers and heads.
    """
    raise NotImplementedError


# --- The run: nothing to change below this line -------------------------------------

# In the tiny model, the checks run on this text, whatever TEXT is: 9 tokens,
# so that code written for 6 tokens or 64 numbers shows its mistake.
CHECK_TEXT: str = "the dog sat on the mat because it"
# Scores far beyond GPT-2's: the score 2000 is in the future and must get 0.
LARGE_SCORES: Array = np.array([[1000.0, 2000.0], [5.0, 5.0]], dtype=np.float32)


def listed(rows: list[int]) -> str:
    """Row numbers as a short text."""
    return " ".join(str(row) for row in rows) if rows else "none"


def check_functions(model: Model, text: str, name: str) -> bool:
    """Check your functions on one head of one model; True if find_head works."""
    ids = checks.encode(model, text)
    t = len(ids)
    trace = forward(ids, model.weights, model.config)
    layer = trace.layers[LAYER]
    q, k, v, s = layer.q[HEAD], layer.k[HEAD], layer.v[HEAD], layer.scores[HEAD]
    d_h = q.shape[1]
    print(f"  {name}, {text!r} ({t} tokens), layer {LAYER}, head {HEAD}:")

    # Task 3: the last row of the head's result, from the last query alone.
    raw = q[-1] @ k.T
    checks.check(
        "newest_row",
        newest_row,
        q[-1],
        k,
        v,
        expected=layer.head_outputs[HEAD][-1],
        mistakes={
            "divide the scores by the square root of d_h": softmax(raw) @ v,
            "take d_h from the shape of q_new: 4 in the tiny model, 64 in GPT-2": (
                softmax(raw / 8) @ v
            ),
            "divide by the square root of d_h, not by d_h": softmax(raw / d_h) @ v,
            "divide by the square root of d_h, a query's length, not of T": (
                softmax(raw / math.sqrt(t)) @ v
            ),
            "the weights must sum to 1: divide each exp() by their sum": (
                np.exp(raw / math.sqrt(d_h)) @ v
            ),
            "the result adds up value rows, not key rows": (
                softmax(raw / math.sqrt(d_h)) @ k
            ),
        },
    )

    # Task 4: all rows at once, with the mask.
    checks.check(
        "attention_weights",
        attention_weights,
        s,
        expected=layer.weights[HEAD],
        mistakes=mask_mistakes(s),
    )

    if not checks.is_tiny(model):
        return False
    checks.check(
        "large scores",
        attention_weights,
        LARGE_SCORES,
        expected=np.array([[1.0, 0.0], [0.5, 0.5]]),
        mistakes=mask_mistakes(LARGE_SCORES),
        if_nan="Subtracting the same number from a row changes no weight: subtract "
        "each row's largest visible score before np.exp, as softmax() in gpt2.py does",
    )

    # Task 6: in the tiny model, the last token, "it", and the token "dog".
    reader, read = t - 1, 2
    weights = trace.attention[:, :, reader, read]  # [layers, heads]
    n_best, h_best = (int(i) for i in np.unravel_index(weights.argmax(), weights.shape))

    def diagnose(mine: Array) -> str:
        n, h = (int(np.rint(value)) for value in mine)
        if (n, h) == (h_best, n_best):
            return "return (layer, head), in this order"
        if not (0 <= n < weights.shape[0] and 0 <= h < weights.shape[1]):
            return f"({n}, {h}) is no head: layers and heads go from 0 to 11"
        return (
            f"in your (layer, head) = ({n}, {h}), A[reader, read] is "
            f"{weights[n, h]:.2f}, but another head gives {weights[n_best, h_best]:.2f}"
            ": compare A[reader, read], row reader and column read, in every head"
        )

    found = checks.check(
        "find_head",
        find_head,
        trace,
        reader,
        read,
        expected=np.array([n_best, h_best]),
        diagnose=diagnose,
    )
    return bool(found)


def mask_mistakes(s: Array) -> dict[str, Array]:
    """What typical mistakes in attention_weights() give for the scores s."""
    t = len(s)
    future = show.future(t)
    masked = s + causal_mask(t)
    by_column = np.exp(masked - masked.max(axis=0, keepdims=True))
    with np.errstate(invalid="ignore"):  # a row with nothing to read gives nan
        diagonal_hidden = softmax(np.where(future | np.eye(t, dtype=bool), -np.inf, s))
    return {
        "hide the future: without a mask, each token reads later ones too": softmax(s),
        "hide the future before the softmax: each row must still sum to 1": (
            softmax(s) * ~future
        ),
        "the future is above the diagonal, where column j > row i": softmax(
            s + causal_mask(t).T
        ),
        "a token may read itself: hide only the cells above the diagonal": (
            diagonal_hidden
        ),
        "the future needs the score -inf, not 0: a score of 0 still gets weight": (
            softmax(np.where(future, 0, s))
        ),
        "softmax within each row (axis=-1), not within each column": (
            by_column / by_column.sum(axis=0, keepdims=True)
        ),
    }


def main() -> None:
    """Show the head for TEXT, compare it with COMPARE, then check your functions."""
    show.utf8_output()
    small = None if USE_TINY else checks.gpt2_small()
    model = small if small is not None else checks.tiny_model()
    config = model.config
    if not (0 <= LAYER < config.n_layers and 0 <= HEAD < config.n_heads):
        raise SystemExit(f"LAYER and HEAD go from 0 to {config.n_layers - 1}.")
    if small is not None:
        print(f"GPT-2 small, layer {LAYER}, head {HEAD}")
    else:
        why = "USE_TINY is True" if USE_TINY else "GPT-2 small is not downloaded"
        print(
            f"The tiny GPT-2 with random weights ({why}), layer {LAYER}, head {HEAD}. "
            "What its heads read means nothing, but every rule of attention holds."
        )

    # Step 1: the head for TEXT, as head() above computes it.
    ids = checks.encode(model, TEXT)
    labels = model.entries(ids)
    t = len(ids)
    trace = forward(ids, model.weights, config)
    layer = trace.layers[LAYER]
    s, a, out = head(layer.q[HEAD], layer.k[HEAD], layer.v[HEAD])
    print(f"\nTEXT, {TEXT!r} with <|endoftext|> in front, is {t} tokens:")
    print("  " + "   ".join(f"{i} {label}" for i, label in enumerate(labels)))
    print("\nS, the scores; row i: token i's query against every key")
    show.print_matrix(s, labels)
    print("\nA, the weights; row i: how token i spreads its attention")
    # The dots mark the future, but only while head() gives it no weight.
    show.print_matrix(a, labels, hide_future=bool(np.all(a[show.future(t)] == 0)))
    print("  Row sums: " + " ".join(f"{total:.2f}" for total in a.sum(axis=-1)))
    print(f"\nV, the values, the first 4 of their {out.shape[1]} numbers per token")
    show.print_matrix(layer.v[HEAD][:, :4], labels)
    print(f"\nThe head's result, the first 4 of its {out.shape[1]} numbers per token")
    show.print_matrix(out[:, :4], labels)
    same = all(
        np.allclose(mine, model_value[HEAD], atol=1e-5)
        for mine, model_value in (
            (s, layer.scores),
            (a, layer.weights),
            (out, layer.head_outputs),
        )
    )
    print(f"  head() computes what the model computes: {same}")

    # Step 2: the same head for COMPARE. Which rows of A are the same?
    ids2 = checks.encode(model, COMPARE)
    layer2 = forward(ids2, model.weights, config).layers[LAYER]
    _, a2, _ = head(layer2.q[HEAD], layer2.k[HEAD], layer2.v[HEAD])
    shared = min(t, len(ids2))
    same_rows = [
        i
        for i in range(shared)
        if np.allclose(a[i, :shared], a2[i, :shared], atol=1e-5)
        and np.allclose(a[i, shared:], 0, atol=1e-5)
        and np.allclose(a2[i, shared:], 0, atol=1e-5)
    ]
    other_rows = [i for i in range(shared) if i not in same_rows]
    labels2 = model.entries(ids2)
    print(f"\nCOMPARE, {COMPARE!r} with <|endoftext|> in front, is {len(ids2)} tokens:")
    print("  " + "   ".join(f"{i} {label}" for i, label in enumerate(labels2)))
    print(f"  Rows of A that are the same as for TEXT: {listed(same_rows)}")
    print(f"  Rows of A that differ: {listed(other_rows)}")
    if len(ids2) != t:
        longer = "COMPARE" if len(ids2) > t else "TEXT"
        last = max(t, len(ids2)) - 1
        rows = f"Row {last}" if last == shared else f"Rows {shared} to {last}"
        print(f"  {rows}: only in {longer}")

    # Step 3: your functions, checked against the forward pass in gpt2.py.
    print("\nYour functions, checked against the forward pass in gpt2.py")
    tiny = model if checks.is_tiny(model) else checks.tiny_model()
    found = check_functions(tiny, CHECK_TEXT, "The tiny GPT-2")
    if small is not None:
        check_functions(small, TEXT, "GPT-2 small")
    elif USE_TINY:
        print("  GPT-2 small: not used, because USE_TINY is True")
    else:
        print("  GPT-2 small: not downloaded, so only the tiny model checked your code")

    # Task 6: once find_head works, it searches the model in use for TEXT.
    if found and not 0 <= FIND < t - 1:
        print(f"\nTask 6: FIND goes from 0 to {t - 2} for this TEXT.")
    elif found:
        n, h = find_head(trace, t - 1, FIND)
        weight = trace.layers[n].weights[h][t - 1, FIND]
        print(
            f"\nTask 6: the last token, {labels[-1]}, gives {labels[FIND]} (FIND = "
            f"{FIND}) the most weight in layer {n}, head {h}: {weight:.2f}"
        )


if __name__ == "__main__":
    main()
