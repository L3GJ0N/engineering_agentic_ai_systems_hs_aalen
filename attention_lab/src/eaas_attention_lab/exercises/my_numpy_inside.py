"""Exercise: Inside NumPy, written and checked by you.

The task sheet is exercises/numpy_inside/README.md in the repository folder.
This file holds everything you change: seven functions to write, plus one for
a star task. Five compute what the header of an array holds or draw random
weights; two compute the scores of one head, S = Q K^T, in ways that
uv run numpy-timings does not show. Each run checks them and prints one line
per function: True, "not written yet", or False with a hint at what to change.
Once a function of part D works, the run also times it for one head of GPT-2
small, T = 1024 tokens, against one @.

The checks compare your results with what NumPy itself computes. Everything
runs offline, with random numbers, without any download.

Run from the attention_lab folder: uv run my-numpy-inside
"""

import math
from collections.abc import Callable

import numpy as np

from eaas_attention_lab import checks, show
from eaas_attention_lab.commands import numpy_timings
from eaas_attention_lab.gpt2 import Array, softmax

# --- Part A: the header of an array --------------------------------------------------


def c_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Task A3: the strides that NumPy gives a new array of this shape, in bytes.

    A new array lies in its block row by row: along the last axis, the next
    number is itemsize bytes further, and one step along any other axis skips
    everything that the axes after it hold. Return one stride per axis, as a
    tuple of ints. Plain Python, no NumPy.
    """
    raise NotImplementedError


def byte_offset(strides: tuple[int, ...], index: tuple[int, ...]) -> int:
    """Task A4: how many bytes after the first number of an array a[index] lies.

    strides comes from the array's header, index holds one position per axis,
    such as (1, 2) for a[1, 2]. The strides of a view may be negative. Plain
    Python, no NumPy.
    """
    raise NotImplementedError


# --- Part B: making arrays -----------------------------------------------------------


def init_head(seed: int, d: int, d_h: int) -> tuple[Array, Array, Array]:
    """Task B3: W_Q, W_K and W_V of one head at random, [d, d_h] each, float32.

    Draw them from a normal distribution with mean 0 and standard deviation
    0.02, the initializer_range in GPT-2's configuration, with one generator
    made from seed. The same seed gives the same three matrices, another seed
    other ones, and the three differ from each other.
    """
    raise NotImplementedError


# --- Part C: views -------------------------------------------------------------------


def transposed(
    shape: tuple[int, ...], strides: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Task C3: the header of a.transpose(axes), from the header of a.

    Axis i of the result is axis axes[i] of a. A transpose writes only a new
    header: return its shape and its strides, two tuples. Plain Python, no
    NumPy.
    """
    raise NotImplementedError


def is_contiguous(
    shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> bool:
    """Task C4: whether this header reads its numbers row by row without a gap.

    That is what NumPy calls C-contiguous, a.flags["C_CONTIGUOUS"]: the strides
    are those of a new array of this shape. NumPy never steps along an axis of
    length 1, so the stride of such an axis does not count. Plain Python, no
    NumPy.
    """
    raise NotImplementedError


# --- Part D: two more ways to compute S = Q K^T ---------------------------------------


def dot_python(x: list[float], y: list[float]) -> float:
    """Task D2: the dot product of two lists of floats, in plain Python.

    x[0] * y[0] + x[1] * y[1] + ... for lists of the same length. No NumPy.
    """
    raise NotImplementedError


def scores_columns(q: Array, k: Array) -> Array:
    """Task D3: S = Q K^T with one @ per column: column j of S is Q k_j.

    q and k are [T, d_h], float32. A Python loop over the T columns of S, each
    computed with one @ and written into its column; return S, [T, T], float32.
    """
    raise NotImplementedError


def softmax_python(row: list[float]) -> list[float]:
    """Part D, star task (optional): the softmax of one row of scores, in Python.

    The weights are exp(x) divided by their sum. Like softmax() in gpt2.py,
    subtract the row's largest number first, so that exp cannot overflow; a
    score of -inf, where the mask hides a token, gets the weight 0. Use the math
    module, no NumPy.
    """
    raise NotImplementedError


# --- The run: nothing to change below this line -------------------------------------

T_CHECK: int = 9  # the tokens of the checks: as many as the check text of my_numpy.py
D_H_CHECK: int = 4  # the width of one head of the tiny GPT-2


def new_strides(
    shape: tuple[int, ...], itemsize: int, order: str = "C"
) -> tuple[int, ...]:
    """The strides of a new array, read from NumPy: order "C" row by row, "F" not."""
    if order == "F":
        return np.empty(shape, dtype=f"u{itemsize}", order="F").strides
    return np.empty(shape, dtype=f"u{itemsize}").strides


def c_strides_cases() -> list[checks.Case]:
    """Shapes of the slides and of GPT-2, with the hints of three typical mistakes."""
    cases = []
    for shape, itemsize, hint in (
        ((2, 3), 8, "float64, 8 bytes per number, rows of 3 numbers"),
        ((1024, 768), 4, "GPT-2 small's position table: float32, rows of 768 numbers"),
        (
            (12, 6, 4),
            4,
            "a stack of 12 heads: a step along axis 0 skips a 6 x 4 matrix",
        ),
        ((5,), 2, "one axis: the next number is one number further"),
        ((2, 3, 4, 5), 1, "four axes, one byte per number"),
        ((3, 1, 4), 4, "an axis of length 1 gets a stride too"),
    ):
        right = new_strides(shape, itemsize)
        mistakes: dict[str, object] = {
            "that is column by column, as in Fortran: a new NumPy array lies row by "
            "row, so the last axis has the smallest stride": new_strides(
                shape, itemsize, "F"
            ),
            "strides count bytes, not numbers: multiply by itemsize": tuple(
                stride // itemsize for stride in right
            ),
            "a step along an axis skips what the axes after it hold, not that axis "
            "itself": tuple(
                stride * length for stride, length in zip(right, shape, strict=True)
            ),
        }
        cases.append(checks.Case((shape, itemsize), right, hint, mistakes))
    return cases


def address(a: Array) -> int:
    """Where the first number of an array lies in memory."""
    return int(a.__array_interface__["data"][0])


def where_numpy_finds(a: Array, index: tuple[int, ...]) -> int:
    """The bytes from a's first number to a[index], as NumPy itself finds them."""
    cell = a[tuple(slice(i, i + 1) for i in index)]
    return address(cell) - address(a)


def byte_offset_cases() -> list[checks.Case]:
    """Arrays of the slides and of gpt2.py, views among them."""
    qkv = np.empty((6, 144), dtype=np.float32)  # like qkv of the tiny GPT-2, 6 tokens
    block = np.empty((3, 4), dtype=np.float32)
    cases = []
    for a, index, hint in (
        (
            np.empty((2, 3)),
            (1, 2),
            "a[1, 2]: one row down and two numbers to the right",
        ),
        (
            np.empty((12, 6, 4), dtype=np.float32),
            (4, 2, 3),
            "three axes: a step along each, each with its own stride",
        ),
        (
            qkv[:, 64:68],
            (5, 3),
            "the keys of head 4 are a window into qkv: their rows lie 576 bytes apart",
        ),
        (block[::-1], (2, 1), "the rows run backwards: a negative stride steps back"),
        (block.T, (3, 2), "after .T, axis 0 steps 4 bytes and axis 1 steps 16"),
    ):
        cases.append(checks.Case((a.strides, index), where_numpy_finds(a, index), hint))
    return cases


def transposed_cases() -> list[checks.Case]:
    """Transposes of the slides and of gpt2.py: split_heads and the scores."""
    stack = np.empty((6, 144), dtype=np.float32)[:, :48].reshape(6, 12, 4)
    heads = stack.transpose(1, 0, 2)  # what split_heads returns
    cases = []
    for a, axes, hint in (
        (np.empty((2, 3)), (1, 0), ".T of a matrix: rows become columns"),
        (
            stack,
            (1, 0, 2),
            "split_heads of gpt2.py: the tokens and the heads swap places",
        ),
        (
            heads,
            (0, 2, 1),
            "the keys in the scores of gpt2.py: only the last two axes swap",
        ),
        (
            np.empty((2, 3, 4, 5)),
            (3, 1, 0, 2),
            "axis i of the result is axis axes[i] of a",
        ),
        (
            np.empty((3, 4), dtype=np.float32)[::-1],
            (1, 0),
            "a negative stride moves with its axis",
        ),
    ):
        result = a.transpose(axes)
        backwards = a.transpose(tuple(int(i) for i in np.argsort(axes)))
        mistakes: dict[str, object] = {
            "the strides move with their axes": (result.shape, a.strides),
            "axes lists the old axes in their new order: axis i of the result is "
            "axis axes[i] of a, not the other way round": (
                backwards.shape,
                backwards.strides,
            ),
        }
        cases.append(
            checks.Case(
                (a.shape, a.strides, axes),
                (result.shape, result.strides),
                hint,
                mistakes,
            )
        )
    return cases


def is_contiguous_cases() -> list[checks.Case]:
    """Fresh arrays and views; the last one has an axis of length 1."""
    cases = []
    for a, hint in (
        (np.empty((2, 3)), "a new array lies row by row"),
        (np.empty((2, 3)).T, "after .T the block is read column by column"),
        (
            np.empty((6, 144), dtype=np.float32)[:, 64:68],
            "a window: after 4 numbers, the next row starts 576 bytes further",
        ),
        (np.empty((12, 6, 4), dtype=np.float32)[3], "one head of a new stack"),
        (np.empty((3, 4), dtype=np.float32)[::-1], "the rows run backwards"),
        (
            np.empty((6, 12, 4), dtype=np.float32).transpose(1, 0, 2),
            "split_heads: the rows of a head are 192 bytes apart, not 16",
        ),
        (
            np.empty((4, 3), dtype=np.float32)[:, np.newaxis, :],
            "np.newaxis adds an axis of length 1 with the stride 0: NumPy never "
            "steps along it, so its stride does not count",
        ),
    ):
        cases.append(
            checks.Case(
                (a.shape, a.strides, a.itemsize), bool(a.flags["C_CONTIGUOUS"]), hint
            )
        )
    return cases


def as_three(call: object) -> list[Array] | None:
    """The three matrices of one call of init_head(), or None if it gave no three."""
    if isinstance(call, tuple | list) and len(call) == 3:
        return [np.asarray(m) for m in call]
    return None


def init_problem(calls: list[object], d: int, d_h: int) -> str:
    """What is wrong with init_head(0), init_head(1), init_head(0), or "" if nothing."""
    found = [as_three(call) for call in calls]
    first, other, again = found
    if first is None or other is None or again is None:
        return "return the three matrices W_Q, W_K and W_V, as a tuple"
    shapes = sorted({m.shape for m in first + other + again})
    if shapes != [(d, d_h)]:
        return f"each matrix is [d, d_h], here ({d}, {d_h}); yours: {shapes}"
    if any(m.dtype != np.float32 for m in first + other + again):
        return "the weights of GPT-2 are float32: make the matrices float32"
    if not all(np.array_equal(a, b) for a, b in zip(first, again, strict=True)):
        return (
            "the same seed must give the same numbers, whatever ran before: make the "
            "generator from seed, inside the function"
        )
    if all(np.array_equal(a, b) for a, b in zip(first, other, strict=True)):
        return "another seed must give other numbers: make the generator from seed"
    w_1, w_2, w_3 = first
    if np.array_equal(w_1, w_2) or np.array_equal(w_2, w_3):
        return (
            "the three matrices are equal: draw all three from one generator, whose "
            "draws continue its sequence"
        )
    numbers = np.concatenate([m.ravel() for m in first])
    if numbers.min() >= 0:
        return "all numbers are positive: draw from a normal distribution around 0"
    spread = float(numbers.std())
    if abs(spread - 0.02) > 0.001:
        if abs(spread - 1) < 0.05:
            return "the standard deviation is 1: scale the draws to 0.02"
        if abs(spread - 0.0004) < 0.0001:
            return "0.02 is the standard deviation, not the variance"
        return f"the standard deviation is {spread:.4f}, not 0.02"
    if abs(float(numbers.mean())) > 0.001:
        return f"the mean is {float(numbers.mean()):.4f}, not 0"
    return ""


def check_init_head(d: int, d_h: int) -> bool | None:
    """Task B3: seed 0, seed 1, seed 0 again; the matrices and their numbers."""
    label = f"    {'init_head:':<22}"
    try:
        calls: list[object] = [init_head(seed, d, d_h) for seed in (0, 1, 0)]
    except NotImplementedError:
        print(label + "not written yet")
        return None
    except Exception as error:  # any error of your code is yours to see
        print(label + f"False, it raised {type(error).__name__}: {error}")
        return False
    problem = init_problem(calls, d, d_h)
    print(label + ("True" if not problem else f"False, {problem}"))
    return not problem


def timed(work: Callable[[], object], what: str, others: dict[str, float]) -> None:
    """Time a function that works, for T = 1024, against other ways (name: seconds)."""
    seconds, _ = numpy_timings.fastest(work)
    against = ", ".join(
        f"{numpy_timings.times_slower(seconds, other)} as long as {name} "
        f"({numpy_timings.duration(other)})"
        for name, other in others.items()
    )
    print(f"{'':26}{what}, T = {numpy_timings.TOKENS}:")
    print(f"{'':26}{numpy_timings.duration(seconds)}: {against}")


def python_pairs(q: list[list[float]], k: list[list[float]]) -> Callable[[], object]:
    """S of one head with dot_python(), one call per pair of tokens."""
    return lambda: [[dot_python(q_i, k_j) for k_j in k] for q_i in q]


def python_rows(s: list[list[float]]) -> Callable[[], object]:
    """The softmax of every row of S with softmax_python()."""
    return lambda: [softmax_python(row) for row in s]


def softmax_cases() -> list[checks.Case]:
    """Three rows: small scores, scores that overflow exp, and masked ones."""
    cases = []
    for row, hint in (
        ([2.0, 1.0, 0.1], "exp of each score, divided by the sum of them all"),
        (
            [1000.0, 999.0, 998.0],
            "exp(1000) is too large for a float: subtract the row's largest number "
            "first",
        ),
        ([0.3, 1.2, -math.inf, -math.inf], "a masked score, -inf, gets the weight 0"),
    ):
        scores = np.array(row)
        mistakes: dict[str, object] = {
            "divide by the sum, so that the weights sum to 1": np.exp(
                scores - scores.max()
            ).tolist()
        }
        expected = softmax(scores[np.newaxis, :])[0].tolist()
        cases.append(checks.Case((row,), expected, hint, mistakes))
    return cases


def check_part_d() -> None:
    """Tasks D2 and D3 and the star task, each timed once it works."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((T_CHECK, D_H_CHECK)).astype(np.float32)
    k = rng.standard_normal((T_CHECK, D_H_CHECK)).astype(np.float32)
    exact = q.astype(np.float64) @ k.astype(np.float64).T
    t, d_h = numpy_timings.TOKENS, numpy_timings.D_H
    q_big = rng.standard_normal((t, d_h)).astype(np.float32)
    k_big = rng.standard_normal((t, d_h)).astype(np.float32)
    one_at, _ = numpy_timings.fastest(numpy_timings.matmul_float32(q_big, k_big))

    print("\nPart D: two more ways to compute S = Q K^T")
    x, y = q[0].tolist(), k[1].tolist()
    works = checks.check(
        "dot_python",
        dot_python,
        x,
        y,
        expected=np.array(exact[0, 1]),
        mistakes={
            "multiply the pairs first, then add the products: x[0] * y[0] + x[1] * "
            "y[1] + ...": np.array(sum(x) * sum(y)),
        },
        if_shape="return one number, the sum of the products",
    )
    if works:
        timed(
            python_pairs(q_big.tolist(), k_big.tolist()),
            "S of one head, one call of dot_python() per pair of tokens",
            {"one @": one_at},
        )
    works = checks.check(
        "scores_columns",
        scores_columns,
        q,
        k,
        expected=exact,
        dtype=np.float32,
        if_dtype="np.zeros and np.empty give float64 unless you pass a dtype: make S "
        "float32",
        if_raised={
            "core dimension": "q is [T, d_h] and k[j] is [d_h]: their product is "
            "column j of S"
        },
    )
    if works:
        rows, _ = numpy_timings.fastest(numpy_timings.numpy_rows(q_big, k_big))
        timed(
            lambda: scores_columns(q_big, k_big),
            "S of one head, one @ per column",
            {"one @": one_at, "one @ per row": rows},
        )

    print("\nPart D, star task (optional)")
    works = checks.check_cases(
        "softmax_python", softmax_python, softmax_cases(), close=True
    )
    if works:
        s = (q_big @ k_big.T / math.sqrt(d_h)).astype(np.float64)
        at_once, _ = numpy_timings.fastest(lambda: softmax(s))
        timed(
            python_rows(s.tolist()),
            "the softmax of every row of S of one head",
            {"softmax() of gpt2.py": at_once},
        )


def main() -> None:
    """Check the functions of this file against what NumPy computes."""
    show.utf8_output()
    print("Checked against NumPy itself, on random numbers: no download.")
    print("\nPart A: the header of an array")
    checks.check_cases("c_strides", c_strides, c_strides_cases())
    checks.check_cases("byte_offset", byte_offset, byte_offset_cases())
    print("\nPart B: making arrays")
    check_init_head(768, 64)
    print("\nPart C: views")
    checks.check_cases("transposed", transposed, transposed_cases())
    checks.check_cases("is_contiguous", is_contiguous, is_contiguous_cases())
    check_part_d()


if __name__ == "__main__":
    main()
