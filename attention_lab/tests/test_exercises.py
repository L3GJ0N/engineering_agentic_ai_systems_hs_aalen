"""The exercise starters run without GPT-2 small and never download it.

They are checked here as students get them, with the functions to write still
empty: the solutions are not part of this repository.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from eaas_attention_lab import checks, model_files
from eaas_attention_lab.exercises import my_head, my_numpy


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
