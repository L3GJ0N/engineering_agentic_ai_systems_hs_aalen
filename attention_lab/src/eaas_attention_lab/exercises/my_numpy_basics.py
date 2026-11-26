"""Exercise: NumPy basics, written and checked by you.

The task sheet is exercises/numpy_basics/README.md in the repository folder.
This file holds the six small functions you write. Each run checks them and
prints one line per function: True, "not written yet", or False with a hint at
what to change. The checks use small arrays of numbers: no download.

Run from the attention_lab folder: uv run my-numpy-basics
"""

import functools

import numpy as np

from eaas_attention_lab import checks, show
from eaas_attention_lab.gpt2 import Array

# --- Part A: arrays, shape and dtype ------------------------------------------------


def make_weights(seed: int, rows: int, columns: int) -> Array:
    """Task A2: a weight matrix at random, [rows, columns], float32.

    Draw its numbers from a normal distribution with mean 0 and standard
    deviation 0.02, the spread of GPT-2's initial weights, with a generator made
    from seed: the same seed gives the same matrix, another seed another one.
    """
    raise NotImplementedError


# --- Part B: indexing ---------------------------------------------------------------


def column(m: Array, j: int) -> Array:
    """Task B2: column j of the matrix m, as a column: shape (rows, 1), not (rows,)."""
    raise NotImplementedError


# --- Part C: element-wise or matrix product -----------------------------------------


def result_shape(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> tuple[int, int]:
    """Task C2: the shape of A @ B for two matrices of these shapes, as a tuple.

    Raise a ValueError when the inner sizes do not fit, as @ does. Plain Python,
    no NumPy.
    """
    raise NotImplementedError


# --- Part D: broadcasting, axis and keepdims ----------------------------------------


def normalize_rows(x: Array) -> Array:
    """Task D2: every row of x divided by its own sum, so that each row sums to 1.

    x is [rows, columns], with positive numbers. No loop.
    """
    raise NotImplementedError


def standardize_columns(x: Array, eps: float) -> Array:
    """Task D3: every column of x with mean 0 and standard deviation 1.

    Subtract each column's mean, then divide by the square root of the column's
    variance with eps added under the root: np.sqrt(variance + eps). No loop.
    """
    raise NotImplementedError


# --- Part E: (5,), (5, 1), (1, 5) and (1, 1) ------------------------------------------


def mse(y_true: Array, y_pred: Array) -> float:
    """Task E4: the mean squared error of two arrays that hold the same n numbers.

    Each may come as (n,), (n, 1) or (1, n). Return one number, not an array.
    """
    raise NotImplementedError


# --- The checks: nothing to change below this line --------------------------------
# Their hints are written for wrong answers: read them after you tried, not before.


def label(name: str) -> str:
    """The start of a check line, as in the other exercises."""
    return f"    {name + ':':<22}"


def weights_problem(calls: list[object]) -> str:
    """What is wrong with make_weights() (seeds 0, 1, 0, then 3 x 5), or ""."""
    arrays = [call for call in calls if isinstance(call, np.ndarray)]
    if len(arrays) != len(calls):
        return "return a NumPy array"
    first, other, again, small = arrays
    if first.shape != (768, 64) or small.shape != (3, 5):
        return (
            f"the matrix is [rows, columns]: make_weights(0, 3, 5) gave {small.shape}"
        )
    if first.dtype != np.float32:
        return "the weights of GPT-2 are float32: make the matrix float32"
    if not np.array_equal(first, again):
        return (
            "the same seed must give the same numbers, whatever ran before: make the "
            "generator from seed, inside the function"
        )
    if np.array_equal(first, other):
        return "another seed must give other numbers: make the generator from seed"
    if first.min() >= 0:
        return "all numbers are positive: draw from a normal distribution around 0"
    mean = float(first.mean())
    if abs(mean) > 0.002:
        return f"the mean is {mean:.4f}, not 0: draw around the mean 0"
    spread = float(first.std())
    if abs(spread - 0.02) > 0.001:
        if abs(spread - 1) < 0.05:
            return "the standard deviation is 1: scale the draws to 0.02"
        if abs(spread - 0.0004) < 0.0001:
            return "0.02 is the standard deviation, not the variance"
        return f"the standard deviation is {spread:.4f}, not 0.02"
    return ""


def check_make_weights() -> None:
    """Task A2: seed 0, seed 1, seed 0 again, and the order of rows and columns."""
    try:
        calls: list[object] = [
            make_weights(0, 768, 64),
            make_weights(1, 768, 64),
            make_weights(0, 768, 64),
            make_weights(0, 3, 5),
        ]
    except NotImplementedError:
        print(label("make_weights") + "not written yet")
        return
    except Exception as error:  # any error of your code is yours to see
        hint = ""
        if isinstance(error, TypeError) and "dtype" in str(error):
            hint = ": normal() has no dtype; draw first, then make the matrix float32"
        print(
            label("make_weights")
            + f"False, it raised {type(error).__name__}: {error}{hint}"
        )
        return
    problem = weights_problem(calls)
    print(label("make_weights") + ("True" if not problem else f"False, {problem}"))


class _Raised:
    """What shape_or_error() gives for a ValueError; the check line shows it so."""

    def __repr__(self) -> str:
        return "a ValueError"


RAISED = _Raised()


@functools.wraps(result_shape)
def shape_or_error(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> object:
    """result_shape(), with a ValueError turned into RAISED."""
    try:
        return result_shape(a_shape, b_shape)
    except ValueError:
        return RAISED


def check_result_shape() -> None:
    """Task C2: shapes from NumPy itself, then the error for sizes that do not fit."""
    cases = []
    for a, b in (
        ((2, 3), (3, 4)),
        ((5, 64), (64, 5)),
        ((1, 5), (5, 1)),
        ((4, 1), (1, 4)),
    ):
        right = (np.zeros(a) @ np.zeros(b)).shape  # what NumPy computes
        mistakes: dict[str, object] = {
            "the inner sizes meet and vanish: the rows of A and the columns of B "
            "remain": (a[1], b[0]),
            "raise a ValueError only when the inner sizes differ": RAISED,
        }
        cases.append(checks.Case((a, b), right, "", mistakes))
    cases.append(
        checks.Case(
            ((2, 3), (2, 3)),
            RAISED,
            "raise a ValueError when the inner sizes differ, as @ does",
        )
    )
    checks.check_cases("result_shape", shape_or_error, cases)


def by_rows(x: Array) -> Array:
    """Each row divided by its own sum, row by row: what normalize_rows() must give."""
    return np.array([[value / sum(row) for value in row] for row in x.tolist()])


def check_normalize_rows() -> None:
    """Task D2, on a square matrix, where a missing keepdims gives no error."""
    x = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 4.0], [2.0, 2.0, 6.0]])
    n = len(x)
    as_row = np.array([[x[i, j] / x[j].sum() for j in range(n)] for i in range(n)])
    by_column = np.array(
        [[x[i, j] / x[:, j].sum() for j in range(n)] for i in range(n)]
    )
    total = float(sum(value for row in x.tolist() for value in row))
    checks.check(
        "normalize_rows",
        normalize_rows,
        x,
        expected=by_rows(x),
        mistakes={
            "the sums of the rows are (3,), and every row is divided by this (3,), "
            "as a row: keep the reduced axis with keepdims=True, so that the sums "
            "are a column (3, 1)": as_row,
            "each row by the sum of its own row, not of its column: sum along "
            "axis 1, the last one": by_column,
            "each row by its own sum, not by the sum of the whole matrix": x / total,
        },
        if_raised={
            "could not be broadcast": "keep the reduced axis with keepdims=True, so "
            "that the sums of the rows, (rows, 1), fit the rows"
        },
    )


def by_columns(x: Array, eps: float, under_root: bool = True) -> Array:
    """Each column standardized on its own, column by column."""
    columns = []
    for c in x.T:
        spread = np.sqrt(c.var() + eps) if under_root else np.sqrt(c.var()) + eps
        columns.append((c - c.mean()) / spread)
    return np.array(columns).T


def check_standardize_columns() -> None:
    """Task D3, with a column whose numbers are all equal."""
    x = np.array([[1.0, 4.0, 2.0], [3.0, 8.0, 2.0], [5.0, 0.0, 2.0], [7.0, 4.0, 2.0]])
    eps = 0.01  # large enough to tell np.sqrt(v + eps) from np.sqrt(v) + eps
    checks.check(
        "standardize_columns",
        standardize_columns,
        x,
        eps,
        expected=by_columns(x, eps),
        mistakes={
            "each column, not each row: take the mean and the variance along axis 0": (
                by_columns(x.T, eps).T
            ),
            "divide by the square root of the variance plus eps, the standard "
            "deviation, not by the variance": np.array(
                [(c - c.mean()) / (c.var() + eps) for c in x.T]
            ).T,
            "add eps under the square root, np.sqrt(variance + eps), not after "
            "it": by_columns(x, eps, under_root=False),
        },
        if_nan="Column 2 holds the same number everywhere, so its variance is 0: add "
        "eps to the variance before the square root",
    )


def check_mse() -> None:
    """Task E4: the same five numbers in four arrangements of shapes."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    guess = truth + 0.1
    for left, right in (
        (truth, guess),
        (truth, guess.reshape(5, 1)),
        (truth.reshape(5, 1), guess),
        (truth.reshape(1, 5), guess.reshape(5, 1)),
    ):
        shapes = f"mse with the shapes {left.shape} and {right.shape}"
        try:
            result = mse(left.copy(), right.copy())
        except NotImplementedError:
            print(label("mse") + "not written yet")
            return
        except Exception as error:  # any error of your code is yours to see
            print(
                label("mse") + f"False, {shapes} raised {type(error).__name__}: {error}"
            )
            return
        if isinstance(result, np.ndarray):
            print(
                label("mse") + f"False, {shapes} returned an array of shape "
                f"{result.shape}: return one number, such as with .item()"
            )
            return
        if not isinstance(result, (float, np.floating)):
            print(label("mse") + f"False, {shapes} returned {type(result).__name__}")
            return
        if abs(float(result) - 0.01) > 1e-9:
            hint = ""
            if abs(float(result) - 4.01) < 1e-9:
                hint = (
                    ": the two shapes broadcast into a 5 x 5 table of differences; "
                    "bring both to one shape first, such as with .ravel()"
                )
            print(
                label("mse")
                + f"False, {shapes} gives {float(result):.4g}, not 0.01{hint}"
            )
            return
    print(label("mse") + "True")


def main() -> None:
    """Check the six functions of this file."""
    show.utf8_output()
    print("Your functions, checked on small arrays: no download.")
    print("\nPart A: arrays, shape and dtype")
    check_make_weights()
    print("\nPart B: indexing")
    m = np.arange(12.0).reshape(3, 4)
    checks.check(
        "column",
        column,
        m,
        2,
        expected=np.array([[row[2]] for row in m.tolist()]),
        if_shape="column 2 holds one number from each of the 3 rows, kept as a "
        "column: (3, 1); a plain index gives (3,)",
    )
    print("\nPart C: element-wise or matrix product")
    check_result_shape()
    print("\nPart D: broadcasting, axis and keepdims")
    check_normalize_rows()
    check_standardize_columns()
    print("\nPart E: (5,), (5, 1), (1, 5) and (1, 1)")
    check_mse()


if __name__ == "__main__":
    main()
