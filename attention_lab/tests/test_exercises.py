"""The exercise starters run without GPT-2 small and never download it.

They are checked here as students get them, with the functions to write still
empty: the solutions are not part of this repository.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from eaas_attention_lab import checks, model_files
from eaas_attention_lab.exercises import my_head


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
