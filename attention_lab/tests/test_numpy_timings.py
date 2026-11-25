"""uv run numpy-timings: nine ways to the same scores, threads, heads, exp, memory.

Run here on a small size and with short budgets, so that the test takes a few
seconds; the times themselves are not checked, only that every way gives the
right numbers and that every part of the printout is there.
"""

import re

import pytest

from eaas_attention_lab.commands import numpy_timings


def test_all_nine_ways_give_the_same_scores(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(numpy_timings, "BUDGET", 0.05)
    numpy_timings.main(["--tokens", "16", "--threads", "1,2"])
    printed = capsys.readouterr().out
    rows = re.findall(
        r"^(\S.{35})\s+([\d,]+)\s+\S+ \S+\s+\S+ x  (True|False)$", printed, re.MULTILINE
    )
    assert [row[0].strip() for row in rows] == [
        name for name, _, _ in numpy_timings.WAYS
    ]
    assert [row[2] for row in rows] == ["True"] * 9
    # Steps in Python for T = 16 and d_h = 64: T x T x d_h, T x T, T, d_h, then 1.
    assert [int(row[1].replace(",", "")) for row in rows] == [
        16 * 16 * 64,
        16 * 16,
        16,
        64,
        1,
        1,
        1,
        1,
        1,
    ]
    assert "with n BLAS threads" in printed
    assert "Twelve heads" in printed
    assert "np.exp on float32" in printed
    assert "GPT-2 small's 124,439,808 weights: 498 MB in float32" in printed


def test_the_threads_run_in_a_fresh_python_that_reads_the_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(numpy_timings, "BUDGET", 0.05)
    numpy_timings.threads(16, [1, 4])
    printed = capsys.readouterr().out
    if "openblas" in numpy_timings.blas().lower():
        assert re.search(r"n = 1 +\S+ \S+", printed)
        assert "(the lab's setting)" in printed
    else:
        assert "does not read OPENBLAS_NUM_THREADS" in printed


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [
        (2.04, "2.04 s"),
        (0.0882, "88.2 ms"),
        (0.000868, "0.868 ms"),
        (1.97e-5, "19.7 µs"),
    ],
)
def test_times_have_three_digits_and_a_unit(seconds: float, shown: str) -> None:
    assert numpy_timings.duration(seconds) == shown


def test_gpt2_small_has_the_weights_that_model_weights_counts() -> None:
    assert numpy_timings.gpt2_small_weights() == 124_439_808
