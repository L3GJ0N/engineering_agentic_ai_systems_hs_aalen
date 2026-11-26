"""The exercise starters run without GPT-2 small and never download it.

They are checked here as students get them, with the functions to write still
empty: the solutions are not part of this repository.
"""

import importlib
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from eaas_attention_lab import checks, model_files
from eaas_attention_lab.exercises import (
    my_head,
    my_numpy,
    my_numpy_basics,
    my_numpy_inside,
)


@pytest.fixture
def no_gpt2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """GPT-2 small looks missing, and a download attempt fails the test."""

    def refuse(*args: object, **kwargs: object) -> str:
        raise AssertionError("the exercise tried to download GPT-2 small")

    monkeypatch.setattr(model_files, "FOLDER", tmp_path / "missing")
    monkeypatch.setattr(model_files, "hf_hub_download", refuse)
    yield


def test_my_head_runs_on_the_tiny_model_without_downloading(
    no_gpt2: None, capsys: pytest.CaptureFixture[str]
) -> None:
    my_head.main()
    output = capsys.readouterr().out
    assert (
        "The tiny GPT-2 with random weights (GPT-2 small is not downloaded)" in output
    )
    assert "head() computes what the model computes: True" in output
    # Appending a word changes no earlier row: the default COMPARE shows it.
    assert "Rows of A that are the same as for TEXT: 0 1 2 3 4 5" in output
    checked = output.split("Your functions,")[1].splitlines()
    verdicts = [line.split(":", 1)[1].strip() for line in checked if line[:4] == "    "]
    assert verdicts == ["not written yet"] * 4


def test_the_tiny_model_knows_its_words_and_names_the_others() -> None:
    model = checks.tiny_model()
    assert len(checks.TINY_WORDS) == model.config.vocab_size
    ids = checks.encode(model, "the cat sat on it")
    assert model.entries(ids) == ["<|endoftext|>", "the", "cat", "sat", "on", "it"]
    with pytest.raises(SystemExit, match="does not know: sofa"):
        checks.encode(model, "the cat sat on the sofa")


# Weights of three tokens that spread their attention evenly over the past.
EVEN = np.tril(np.ones((3, 3))) / np.arange(1, 4)[:, None]


@pytest.mark.parametrize(
    ("result", "hint"),
    [
        (0.5 * EVEN, "yours is 0.5 times the expected values"),
        (
            np.tril(np.ones((3, 3))),
            "each row must sum to 1; yours sum to 1.00 2.00 3.00",
        ),
        (EVEN[::-1], "the largest difference is at"),
        (np.full((3, 3), np.nan), "nan or inf in row 0, 1, 2"),
        (np.ones((2, 3)), "its shape is (2, 3), expected (3, 3)"),
        (None, "it returned None"),
    ],
)
def test_a_wrong_result_gets_a_hint(
    result: object, hint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert checks.check("f", lambda: result, expected=EVEN) is False
    assert hint in capsys.readouterr().out


def test_a_transposed_result_is_named(capsys: pytest.CaptureFixture[str]) -> None:
    expected = np.arange(9.0).reshape(3, 3)
    assert checks.check("f", lambda: expected.T, expected=expected) is False
    assert "yours is the transpose" in capsys.readouterr().out


def test_the_right_numbers_in_another_dtype_are_named(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = np.arange(4.0, dtype=np.float32)
    result = checks.check(
        "f",
        lambda: expected.astype(np.float64),
        expected=expected,
        dtype=np.float32,
        if_dtype="make it float32",
    )
    assert result is False
    assert "its dtype is float64, not float32: make it float32" in (
        capsys.readouterr().out
    )


def test_an_error_and_a_wrong_shape_get_their_hints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken() -> object:
        raise ValueError("matmul: Input operand 1 has a mismatch in its core dimension")

    hints = {"core dimension": "swap the last two axes"}
    assert checks.check("f", broken, expected=EVEN, if_raised=hints) is False
    assert "core dimension. Swap the last two axes" in capsys.readouterr().out
    assert checks.check("f", lambda: 7, expected=np.ones(2), if_shape="two") is False
    assert "its shape is (), expected (2,): two" in capsys.readouterr().out


def test_my_numpy_runs_on_the_tiny_model_without_downloading(
    no_gpt2: None, capsys: pytest.CaptureFixture[str]
) -> None:
    my_numpy.main()
    output = capsys.readouterr().out
    assert "The tiny GPT-2 with random weights" in output
    lines = [line.strip() for line in output.splitlines() if line[:4] == "    "]
    verdicts = dict(line.split(":", 1) for line in lines)
    # The given function holds the mistake of task A3; every other one is a stub.
    assert (
        verdicts.pop("scaled_scores")
        .strip()
        .startswith("False, the numbers are right, but its dtype is float64")
    )
    assert [verdict.strip() for verdict in verdicts.values()] == ["not written yet"] * 7


def test_tiny_trace_gives_the_stacks_of_the_task_sheet() -> None:
    trace = my_numpy.tiny_trace("the cat sat on it")
    assert trace.attention.shape == (12, 12, 6, 6)
    assert trace.layers[5].q.dtype == np.float32


def test_my_numpy_inside_checks_eight_stubs_and_times_nothing(
    no_gpt2: None, capsys: pytest.CaptureFixture[str]
) -> None:
    my_numpy_inside.main()
    output = capsys.readouterr().out
    lines = [line.strip() for line in output.splitlines() if line[:4] == "    "]
    verdicts = [line.split(":", 1)[1].strip() for line in lines]
    assert verdicts == ["not written yet"] * 8
    assert "T = 1024" not in output


def test_the_cases_of_my_numpy_inside_come_from_numpy() -> None:
    # The expected results are what NumPy computes, never a formula of the check.
    strides = my_numpy_inside.c_strides_cases()
    assert strides[0].args == ((2, 3), 8)
    assert strides[0].expected == (24, 8)
    offsets = my_numpy_inside.byte_offset_cases()
    assert [case.expected for case in offsets] == [40, 428, 652, -28, 44]
    window = my_numpy_inside.is_contiguous_cases()[-1]
    assert window.args == ((4, 1, 3), (12, 0, 4), 4)
    assert window.expected is True


def test_my_numpy_basics_checks_six_stubs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    my_numpy_basics.main()
    output = capsys.readouterr().out
    lines = [line.strip() for line in output.splitlines() if line[:4] == "    "]
    verdicts = [line.split(":", 1)[1].strip() for line in lines]
    assert verdicts == ["not written yet"] * 6


@pytest.mark.parametrize(
    ("name", "mistake", "hint"),
    [
        ("column", lambda m, j: m[:, j], "kept as a column"),
        ("normalize_rows", lambda x: x / x.sum(axis=1), "keepdims=True"),
        ("mse", lambda a, b: float(((a - b) ** 2).mean()), "5 x 5 table"),
        ("mse", lambda a, b: np.ones((1, 1)), "return one number"),
    ],
)
def test_typical_mistakes_of_my_numpy_basics_get_their_hints(
    name: str,
    mistake: Callable[..., object],
    hint: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Mistakes only: the solutions are not part of this repository.
    monkeypatch.setattr(my_numpy_basics, name, mistake)
    my_numpy_basics.main()
    output = capsys.readouterr().out
    line = next(x for x in output.splitlines() if x.startswith(f"    {name}:"))
    verdict = line.split(":", 1)[1].strip()
    assert verdict.startswith("False")
    assert hint in verdict


def square(x: int) -> int:
    """A function to check case by case: right for 2, wrong for 3."""
    return 4 if x == 2 else 10


@pytest.mark.parametrize(
    ("cases", "line"),
    [
        ([checks.Case((2,), 4)], "f:                    True"),
        (
            [checks.Case((2,), 4), checks.Case((3,), 9, "three times three")],
            "False, f(3) gives 10, not 9: three times three",
        ),
        (
            [checks.Case((3,), 9, "three times three", {"one too many": 10})],
            "False, f(3) gives 10, not 9: one too many",
        ),
    ],
)
def test_check_cases_shows_the_first_call_that_differs(
    cases: list[checks.Case], line: str, capsys: pytest.CaptureFixture[str]
) -> None:
    checks.check_cases("f", square, cases)
    assert line in capsys.readouterr().out


def test_check_cases_names_stubs_errors_and_rounding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def stub(x: int) -> int:
        raise NotImplementedError

    def broken(x: int) -> int:
        raise ZeroDivisionError("division by zero")

    assert checks.check_cases("f", stub, [checks.Case((1,), 1)]) is None
    assert "not written yet" in capsys.readouterr().out
    assert checks.check_cases("f", broken, [checks.Case((1,), 1, "x is 1")]) is False
    assert "f(1) raised ZeroDivisionError: division by zero: x is 1" in (
        capsys.readouterr().out
    )
    almost = [checks.Case((1,), [0.1 + 0.2])]
    assert checks.check_cases("f", lambda x: [0.3], almost) is False
    assert checks.check_cases("f", lambda x: [0.3], almost, close=True) is True


def test_every_command_of_pyproject_has_a_main() -> None:
    # uv run <name> calls these; a missing entry or module would fail only there.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    commands = {"my-numpy-basics", "my-numpy-inside", "numpy-timings", "my-numpy"}
    assert commands | {"my-head"} <= set(scripts)
    for target in scripts.values():
        module, function = target.split(":")
        assert callable(getattr(importlib.import_module(module), function))
