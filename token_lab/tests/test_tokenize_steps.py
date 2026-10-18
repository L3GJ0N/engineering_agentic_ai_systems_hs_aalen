"""The two step-by-step scripts must print the known tokens of the course text.

Runs only after the scripts downloaded the Nemotron files once; the tests hand
the scripts those files instead of letting them use the network. The expected
IDs belong to the pinned revision and were checked against the Transformers
library on 18 October 2026.
"""

import re

import pytest

from eaas_token_lab import tokenize_chat, tokenize_text

FOLDER = tokenize_text.FOLDER
pytestmark = pytest.mark.skipif(
    not all((FOLDER / f).is_file() for f in ("tokenizer.json", "chat_template.jinja")),
    reason="Nemotron tokenizer files not downloaded yet: run uv run tokenize-chat",
)

# "Explain tokenization in one sentence." alone.
TEXT_IDS: list[int] = [2763, 44122, 10835, 3229, 1294, 1925, 19286, 1046]
# The same text after the chat template, with thinking switched off.
CHAT_IDS: list[int] = [
    *(10, 25708, 1010, 11, 1010),  # <|im_start|>system \n <|im_end|> \n
    *(10, 3263, 1010),  # <|im_start|>user \n
    *TEXT_IDS,
    *(11, 1010),  # <|im_end|> \n
    *(10, 1503, 19464, 1010),  # <|im_start|>assistant \n
    *(12, 13),  # <think></think>
]


def local_file(repo_id: str, filename: str, **options: object) -> str:
    """Stand in for hf_hub_download: return the already downloaded file."""
    return str(FOLDER / filename)


def printed_ids(printed: str) -> list[int]:
    """Read the IDs from token rows such as "  3229  ization"."""
    return [int(m) for m in re.findall(r"^ *(\d+)  \S", printed, re.MULTILINE)]


def test_step_1_shows_the_tokens_of_the_text_alone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tokenize_text, "hf_hub_download", local_file)
    tokenize_text.main()
    printed: str = capsys.readouterr().out
    assert printed_ids(printed) == TEXT_IDS
    assert "gives back the text: True" in printed


def test_step_2_shows_what_the_model_reads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tokenize_chat, "hf_hub_download", local_file)
    tokenize_chat.main()
    printed: str = capsys.readouterr().out
    assert printed_ids(printed) == CHAT_IDS
    assert "gives back the text of step 2: True" in printed
    # The markers are flagged as special, the role name is not.
    assert re.search(r"^ +10  <\|im_start\|>   special token$", printed, re.MULTILINE)
    assert re.search(r"^ +3263  user$", printed, re.MULTILINE)
