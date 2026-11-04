"""The scripts must print the known facts of GPT-2 small and draw their figures.

Runs only after a script downloaded the GPT-2 files once; the tests hand the
scripts those files instead of letting them use the network, and collect the
figures in a temporary folder. The expected numbers belong to the pinned
revision. test_reference.py checks the forward pass itself against Hugging
Face Transformers.
"""

import re
from pathlib import Path

import pytest

from eaas_attention_lab import (
    attention_sink,
    model_files,
    model_weights,
    one_head,
    show,
)

FOLDER = model_files.FOLDER
pytestmark = pytest.mark.skipif(
    not all((FOLDER / name).is_file() for name in model_files.FILES),
    reason="GPT-2 files not downloaded yet: run uv run model-weights",
)


def local_file(repo_id: str, filename: str, **options: object) -> str:
    """Stand in for hf_hub_download: check the call, return the downloaded file.

    The scripts must ask for the pinned revision and never send a login.
    """
    assert repo_id == model_files.REPOSITORY
    assert options == {
        "revision": model_files.REVISION,
        "local_dir": FOLDER,
        "token": False,
    }
    return str(FOLDER / filename)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_files, "hf_hub_download", local_file)
    monkeypatch.setattr(show, "OUTPUTS", tmp_path)


def test_model_weights_finds_the_matrices_and_accounts_for_every_byte(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    model_weights.main()
    printed = capsys.readouterr().out
    assert "holds 160 named arrays" in printed
    assert "4 + 12 x 13 = 160: True" in printed
    assert "4 x 768 x 768 = 2,359,296 weights in these four matrices" in printed
    assert "plus 3,072 biases" in printed
    assert "the same in all 12 blocks: True" in printed
    assert "no matrix for the output layer: True" in printed
    assert "124,439,808 learned weights" in printed
    assert re.search(r"feed-forward networks, 12 blocks +56,669,184 +45\.54%", printed)
    assert "548,105,171 bytes, the size of the file: True" in printed
    assert (tmp_path / "model_weights.png").is_file()


def test_one_head_matches_the_model_at_every_step(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    one_head.main()
    printed = capsys.readouterr().out
    ids = re.findall(r"^ +\d  +(\d+)  \S", printed, re.MULTILINE)
    assert ids == ["50256", "1169", "3797", "3332", "319", "340"]
    # Q, K and V (step 2), S (step 3), A (step 4) and A V (step 7) as in the
    # forward pass.
    assert printed.count("The model computed the same: True") == 4
    assert "Each row sums to 1: True" in printed
    assert "the order within every row stays the same: True" in printed
    assert "rows that sum to 0.19 0.66 0.55 0.55 0.71 1.00" in printed
    assert "gives A of step 4 back: True" in printed
    assert "Ġit gets 0.68 x value(Ġon)" in printed
    assert (tmp_path / "one_head.png").is_file()


def test_attention_sink_shows_the_position_not_the_word(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    attention_sink.main()
    printed = capsys.readouterr().out
    assert "in 114 of 144 heads" in printed
    for first, count in (("the", 116), ("Yesterday", 108), ("So", 113)):
        assert re.search(rf"^  {first} +{count} of 144 heads", printed, re.MULTILINE)
    assert (
        "0.61 of its weight on Ġthe at position 0, and 0.09 on Ġthe at position 4"
        in (printed)
    )
    assert "in the median head, 0.16 times" in printed
    assert "on average 0.29 of the lengths of all tokens' parts added up" in printed
    assert "although it holds 0.58 of the weight" in printed
    # The heads that park their weight return little, those that read return more.
    assert re.search(r"In the 19 heads .* on average 0\.15 times", printed)
    assert re.search(r"in the 20 heads .* there, 0\.78 times", printed)
    assert re.search(
        r"^ +5 +3,029 +78 to 96 +number 447: 2,917, number 138: 811$",
        printed,
        re.MULTILINE,
    )
    assert (tmp_path / "attention_sink_heads.png").is_file()
    assert (tmp_path / "attention_sink_layers.png").is_file()
