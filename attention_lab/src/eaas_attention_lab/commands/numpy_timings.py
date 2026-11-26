"""How fast is NumPy, and why? One computation done nine ways, and timed.

The scores of one attention head, S = Q K^T, are the first step of attention.
For a text of 1,024 tokens, the longest GPT-2 small reads, Q and K of one head
are [1024, 64], and S is [1024, 1024]: 67 million multiplications. This file
computes S in nine ways, from three loops in plain Python to one @, which
NumPy hands to BLAS, a library for matrix products, and prints how long each
way takes on your laptop. Then it times the same @ with different numbers of
BLAS threads, twelve heads in a loop against one @ on their stack, the
element-wise function exp in Python and in NumPy, and what a number costs in
memory as a Python float and in an array.

Every way gets the same random numbers, and every result is compared with
the exact product: the column "Same" says True when they agree up to the
rounding of float32. Each time is the fastest of several runs; a way that takes
longer than a second runs once. Other programs and the power settings of your
laptop change the times; the order of magnitude stays.

Offline, random numbers only, no download. It takes about 15 seconds.

Setup: uv sync
Run from the project root: uv run numpy-timings
Other sizes for the nine ways and the threads: uv run numpy-timings --tokens 256
(1 to 4096 tokens; 4096 takes about a minute and 1.5 GB of memory)
Read next: exercises/numpy_inside/README.md, part E
"""

import argparse
import datetime
import math
import os
import platform
import struct
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eaas_attention_lab import show

Matrix = NDArray[Any]

# --- The settings ----------------------------------------------------------------

TOKENS: int = 1024  # T: the longest text GPT-2 small reads
D_H: int = 64  # the width of one head of GPT-2 small
HEADS: int = 12  # heads per block in GPT-2 small
FEW_TOKENS: int = 9  # a short text, as in the checks of the exercises
BUDGET: float = 1.0  # seconds per way: as many runs as fit, the fastest counts
MOST_RUNS: int = 30
MOST_BYTES: float = 1e9  # the largest array a way may build: 1 GB
MOST_TOKENS: int = 4096  # plain Python grows with T x T: 4096 takes about 30 s
SEED: int = 0

# GPT-2 small: the width d, the blocks, the vocabulary and the positions.
GPT2_SMALL: tuple[int, int, int, int] = (768, 12, 50_257, 1_024)
MODULE = "eaas_attention_lab.commands.numpy_timings"


# --- Measuring ---------------------------------------------------------------------


def fastest(run: Callable[[], object]) -> tuple[float, object]:
    """The fastest of as many runs as fit into BUDGET seconds, and the last result.

    A run that takes longer than BUDGET runs once. The first run of a fast one
    also pays for warming up, such as waking the BLAS threads, so it is rarely
    the fastest.
    """
    times: list[float] = []
    result: object = None
    started = time.perf_counter()
    while len(times) < MOST_RUNS:
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        if time.perf_counter() - started > BUDGET:
            break
    return min(times), result


def duration(seconds: float) -> str:
    """A time with three digits and a unit: 1.94 s, 88.2 ms, 0.868 ms, 19.7 µs."""
    if seconds >= 1:
        return f"{seconds:.3g} s"
    if seconds >= 1e-4:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def times_slower(seconds: float, best: float) -> str:
    """How many times slower than best: 2,230 x, 6.2 x, 1 x."""
    ratio = seconds / best
    return f"{ratio:,.0f} x" if ratio >= 10 else f"{ratio:.1f} x"


# --- Nine ways to compute S = Q K^T ------------------------------------------------
#
# Each function gets Q and K, [T, d_h] in float32, prepares its inputs (not
# timed) and returns the work to time, which returns S.


def python_loops(q: Matrix, k: Matrix) -> Callable[[], object]:
    """Three loops over lists of Python floats: the formula, one product at a time."""
    q_rows, k_rows = q.tolist(), k.tolist()
    t, d_h = q.shape

    def run() -> list[list[float]]:
        s = []
        for i in range(t):
            q_i = q_rows[i]
            row = []
            for j in range(t):
                k_j = k_rows[j]
                total = 0.0
                for c in range(d_h):
                    total += q_i[c] * k_j[c]
                row.append(total)
            s.append(row)
        return s

    return run


def numpy_pairs(q: Matrix, k: Matrix) -> Callable[[], object]:
    """A Python loop over all pairs (i, j); NumPy hands each dot product to BLAS."""
    t = len(q)

    def run() -> Matrix:
        s = np.empty((t, t), dtype=np.float32)
        for i in range(t):
            for j in range(t):
                s[i, j] = q[i] @ k[j]
        return s

    return run


def numpy_rows(q: Matrix, k: Matrix) -> Callable[[], object]:
    """A Python loop over the rows: NumPy hands each row of S, q_i K^T, to BLAS."""
    t = len(q)

    def run() -> Matrix:
        s = np.empty((t, t), dtype=np.float32)
        for i in range(t):
            s[i] = q[i] @ k.T
        return s

    return run


def numpy_outer(q: Matrix, k: Matrix) -> Callable[[], object]:
    """A Python loop over the d_h columns: S as a sum of outer products."""
    t, d_h = q.shape

    def run() -> Matrix:
        s = np.zeros((t, t), dtype=np.float32)
        for c in range(d_h):
            s += np.outer(q[:, c], k[:, c])
        return s

    return run


def broadcasting(q: Matrix, k: Matrix) -> Callable[[], object]:
    """No loop: every product at once in a [T, T, d_h] array, then its sum.

    q[:, np.newaxis, :] is [T, 1, d_h] and k[np.newaxis, :, :] is [1, T, d_h]:
    the axes of length 1 stretch, so that row i of q meets every row of k.
    """
    return lambda: (q[:, np.newaxis, :] * k[np.newaxis, :, :]).sum(axis=-1)


def einsum(q: Matrix, k: Matrix) -> Callable[[], object]:
    """No loop: np.einsum with its own loops in C, without BLAS."""
    return lambda: np.einsum("id,jd->ij", q, k)


def matmul_int32(q: Matrix, k: Matrix) -> Callable[[], object]:
    """One @ on whole numbers, 10 q and 10 k rounded: NumPy's own loops, no BLAS."""
    q_int = np.round(10 * q).astype(np.int32)
    k_int = np.round(10 * k).astype(np.int32)
    return lambda: q_int @ k_int.T


def matmul_float64(q: Matrix, k: Matrix) -> Callable[[], object]:
    """One @ on float64 numbers, 8 bytes each: BLAS."""
    q64, k64 = q.astype(np.float64), k.astype(np.float64)
    return lambda: q64 @ k64.T


def matmul_float32(q: Matrix, k: Matrix) -> Callable[[], object]:
    """One @ on float32 numbers, 4 bytes each, as in gpt2.py: BLAS."""
    return lambda: q @ k.T


# (name, steps of Python loops for T and d_h, the way)
WAYS: list[tuple[str, Callable[[int, int], int], Callable[[Matrix, Matrix], Any]]] = [
    ("Python: three loops over lists", lambda t, d_h: t * t * d_h, python_loops),
    ("NumPy, one dot product per pair", lambda t, d_h: t * t, numpy_pairs),
    ("NumPy, one row per step", lambda t, d_h: t, numpy_rows),
    ("NumPy, one outer product per step", lambda t, d_h: d_h, numpy_outer),
    ("NumPy, broadcasting, then sum", lambda t, d_h: 1, broadcasting),
    ("np.einsum", lambda t, d_h: 1, einsum),
    ("one @ on int32 (no BLAS)", lambda t, d_h: 1, matmul_int32),
    ("one @ on float64 (BLAS)", lambda t, d_h: 1, matmul_float64),
    ("one @ on float32 (BLAS)", lambda t, d_h: 1, matmul_float32),
]


def same(result: object, q: Matrix, k: Matrix, whole: bool) -> bool:
    """Whether a result is the exact product, up to float32 rounding."""
    if whole:
        q_int = np.round(10 * q).astype(np.float64)
        k_int = np.round(10 * k).astype(np.float64)
        return bool(np.array_equal(np.asarray(result), q_int @ k_int.T))
    exact = q.astype(np.float64) @ k.astype(np.float64).T
    return bool(np.allclose(np.asarray(result, dtype=np.float64), exact, atol=1e-3))


def nine_ways(q: Matrix, k: Matrix) -> float:
    """Time and check every way; print the table. Returns the time of one @.

    Broadcasting holds all T x T x d_h products at once; above MOST_BYTES it is
    left out, so that a large --tokens does not fill the memory of a laptop.
    """
    t, d_h = q.shape
    rows: list[tuple[str, int, float | None, bool | None]] = []
    for name, steps, way in WAYS:
        if way is broadcasting and t * t * d_h * q.itemsize > MOST_BYTES:
            rows.append((name, steps(t, d_h), None, None))
            continue
        seconds, result = fastest(way(q, k))
        rows.append((name, steps(t, d_h), seconds, same(result, q, k, "int32" in name)))
    best = rows[-1][2] or 1.0
    print(f"\n{'Way':<36}{'Steps in Python':>16}{'Time':>11}{'vs. one @':>12}  Same")
    for name, count, took, agrees in rows:
        if took is None:
            gigabytes = t * t * d_h * q.itemsize / 1e9
            print(f"{name:<36}{count:>16,}  left out: it needs {gigabytes:.1f} GB")
            continue
        print(
            f"{name:<36}{count:>16,}{duration(took):>11}"
            f"{times_slower(took, best):>12}  {agrees}"
        )
    return best


# --- Threads, heads, an element-wise function and memory ---------------------------


def blas() -> str:
    """The BLAS library of this NumPy and its version, as np.show_config() names it."""
    config: Any = np.show_config(mode="dicts")
    found = config.get("Build Dependencies", {}).get("blas", {})
    return f"{found.get('name', 'unknown')} {found.get('version', '')}".strip()


def one_product(t: int) -> float:
    """The fastest @ of one head's Q and K^T in float32, in seconds."""
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((t, D_H)).astype(np.float32)
    k = rng.standard_normal((t, D_H)).astype(np.float32)
    seconds, _ = fastest(matmul_float32(q, k))
    return seconds


def threads(t: int, counts: Sequence[int]) -> None:
    """One @ with n BLAS threads, each in a fresh Python that reads the setting."""
    print(f"\nOne @ on float32, [{t}, {D_H}] @ [{D_H}, {t}], with n BLAS threads:")
    if "openblas" not in blas().lower():
        print(f"  Your NumPy uses {blas()}, which does not read OPENBLAS_NUM_THREADS.")
        return
    for n in counts:
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": str(n)}
        # sys.executable and fixed arguments: this file, run once more.
        child = subprocess.run(  # noqa: S603
            [sys.executable, "-m", MODULE, "--one-product", str(t)],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        note = "  (the lab's setting)" if n == 4 else ""
        if n == os.cpu_count():
            note = "  (one per logical CPU)"
        print(f"  n = {n:<3}{duration(float(child.stdout)):>11}{note}")


def head_loop(q: Matrix, k: Matrix) -> Callable[[], object]:
    """A Python loop over the heads of a stack [heads, T, d_h]: one @ each."""
    return lambda: [q[h] @ k[h].T for h in range(len(q))]


def head_stack(q: Matrix, k: Matrix) -> Callable[[], object]:
    """One @ on the whole stack: the last two axes of k swapped, as in gpt2.py."""
    return lambda: q @ k.transpose(0, 2, 1)


def heads(t: int) -> None:
    """Twelve heads: a loop of 12 @ against one @ on the stack [12, T, d_h]."""
    rng = np.random.default_rng(SEED)
    print(f"\nTwelve heads, a loop of {HEADS} @ against one @ on their stack:")
    for tokens in (t, FEW_TOKENS):
        q = rng.standard_normal((HEADS, tokens, D_H)).astype(np.float32)
        k = rng.standard_normal((HEADS, tokens, D_H)).astype(np.float32)
        loop, _ = fastest(head_loop(q, k))
        stack, _ = fastest(head_stack(q, k))
        print(
            f"  T = {tokens:<5} the loop {duration(loop):>9}, the stack "
            f"{duration(stack):>9}: the stack is {loop / stack:.1f} x as fast"
        )


def element_wise(t: int) -> None:
    """The exponential of T x T numbers: math.exp in a Python loop, then np.exp."""
    numbers = np.random.default_rng(SEED).standard_normal(t * t)
    as_list = numbers.tolist()
    as_float32 = numbers.astype(np.float32)
    print(f"\nAn element-wise function: exp of {t * t:,} numbers, one head's scores:")
    loop, _ = fastest(lambda: [math.exp(x) for x in as_list])
    exp64, _ = fastest(lambda: np.exp(numbers))
    exp32, _ = fastest(lambda: np.exp(as_float32))
    print(f"  math.exp in a Python loop over a list   {duration(loop):>9}")
    print(f"  np.exp on float64                        {duration(exp64):>9}")
    print(f"  np.exp on float32                        {duration(exp32):>9}")


def gpt2_small_weights() -> int:
    """The learned weights of GPT-2 small, from its shapes: 124,439,808."""
    d, blocks, vocabulary, positions = GPT2_SMALL
    layer_norms = 2 * 2 * d
    attention = d * 3 * d + 3 * d + d * d + d
    feed_forward = d * 4 * d + 4 * d + 4 * d * d + d
    block = layer_norms + attention + feed_forward
    return vocabulary * d + positions * d + blocks * block + 2 * d


def memory() -> None:
    """What a number costs: as a Python float in a list, and in a float32 array."""
    n = 1_000_000
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    floats = [float(i) + 0.5 for i in range(n)]  # a million different objects
    in_list = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    del floats
    array = np.zeros(n, dtype=np.float32).nbytes
    pointer = struct.calcsize("P")
    print("\nMemory:")
    print(
        f"  One Python float: {sys.getsizeof(1.0)} bytes, plus {pointer} bytes for "
        "the list's pointer to it; in a float32 array: 4 bytes"
    )
    print(
        f"  A list of {n:,} different floats: {in_list / 1e6:.1f} MB; "
        f"a float32 array of {n:,}: {array / 1e6:.1f} MB"
    )
    weights = gpt2_small_weights()
    print(
        f"  GPT-2 small's {weights:,} weights: {weights * 4 / 1e6:,.0f} MB in float32, "
        f"at least {weights * in_list / n / 1e9:.1f} GB as a list of Python floats"
    )


def header() -> None:
    """Versions, the BLAS library, the processor and the date of this run."""
    setting = os.environ.get("OPENBLAS_NUM_THREADS", "not set")
    print(
        f"numpy {np.__version__} | python {platform.python_version()} | {blas()}, "
        f"OPENBLAS_NUM_THREADS={setting} | {platform.machine()}, "
        f"{os.cpu_count()} logical CPUs | {datetime.date.today().isoformat()}"
    )


def at_least_one(text: str) -> int:
    """A whole number of at least 1, for --tokens and each count of --threads."""
    number = int(text)  # argparse reports a ValueError as an invalid value
    if number < 1:
        raise argparse.ArgumentTypeError(f"not a whole number of at least 1: {text}")
    return number


def thread_counts(text: str) -> list[int]:
    """The counts of --threads, such as 1,4."""
    return [at_least_one(n) for n in text.split(",") if n.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    """Time the nine ways, then threads, heads, exp and memory."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--tokens",
        type=at_least_one,
        default=TOKENS,
        help="T of the nine ways and of the threads, 1 to 4096, default 1024; the "
        "heads and exp keep 1024. Plain Python takes about 2 s for 1024, and 30 s "
        "and 1.5 GB of memory for 4096",
    )
    parser.add_argument(
        "--threads",
        type=thread_counts,
        default=[],
        help="BLAS thread counts to try, such as 1,4; default 1, 2, 4, 8 and all",
    )
    parser.add_argument("--one-product", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one_product is not None:  # the run of threads() in a fresh Python
        print(one_product(args.one_product))
        return
    if args.tokens > MOST_TOKENS:
        parser.error(
            f"--tokens takes 1 to {MOST_TOKENS}, not {args.tokens}: time and memory "
            "grow with T x T"
        )
    show.utf8_output()
    header()
    t = args.tokens
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((t, D_H)).astype(np.float32)
    k = rng.standard_normal((t, D_H)).astype(np.float32)
    print(
        f"\nThe scores of one head, S = Q K^T: Q and K are [{t}, {D_H}], float32, "
        f"S is [{t}, {t}]"
    )
    nine_ways(q, k)
    cpus = os.cpu_count() or 1
    counts = args.threads or sorted({n for n in (1, 2, 4, 8, cpus) if n <= cpus})
    threads(t, counts)
    heads(TOKENS)
    element_wise(TOKENS)
    memory()


if __name__ == "__main__":
    main()
