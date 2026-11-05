"""GPT-2 small's files: download them once, then read them.

GPT-2 small is the smallest GPT-2, published by OpenAI in 2019 with its
weights. It runs on a laptop CPU, and its design is the plain one that
attention diagrams draw: every head has its own queries, keys and values, and
positions are learned vectors, added once to the input. Its sizes: d = 768,
12 blocks with 12 heads of 64, a vocabulary of 50,257 entries. Current models
are much larger and differ in such details; the README says how.

Three files of its Hugging Face repository are enough:
  config.json        the hyperparameters: d, the number of blocks and heads
  tokenizer.json     the vocabulary and the rules that split a text into it
  model.safetensors  all weights, as named arrays of 32-bit floats, 548 MB
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tokenizers import Tokenizer

from eaas_attention_lab.gpt2 import Array, Config

# The model's repository on Hugging Face: https://huggingface.co/openai-community/gpt2
# Its files are public, and no account is needed. Its model card gives the
# licence as MIT; OpenAI's own GPT-2 repository on GitHub uses a modified MIT
# licence, which also asks you to use GPT-2 responsibly and to say when
# content was created with it.
REPOSITORY: str = "openai-community/gpt2"

# A revision pins one version of the repository's files, so that the numbers
# never change under us. This is its newest commit, checked on 4 November
# 2026 at https://huggingface.co/api/models/openai-community/gpt2 (value
# "sha"); the files have not changed since February 2024. token_lab's
# tokenize_text.py explains revisions in detail.
REVISION: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

FILES: tuple[str, ...] = ("config.json", "tokenizer.json", "model.safetensors")

# Downloaded files are kept as plain files in attention_lab/downloads, where
# you can open them. Every attention_lab command uses the same folder.
FOLDER: Path = (
    Path(__file__).resolve().parents[2] / "downloads" / "gpt2" / REVISION[:12]
)

# GPT-2's only special token. In its training data it separates documents,
# so a text that follows it reads like the start of a new document.
END_OF_TEXT: str = "<|endoftext|>"


def download() -> Path:
    """Fetch the three files (only those not yet there) and return their folder.

    For each file, hf_hub_download asks
    https://huggingface.co/<REPOSITORY>/resolve/<REVISION>/<file>
    and keeps the file. The weights file then comes in pieces from Hugging
    Face's storage: about 475 MB over the network for 548 MB on disk. A file
    that is complete is reused without asking the server again. An
    interrupted download does not continue where it stopped: run the command
    again, and the unfinished file starts over from zero.
    """
    for name in FILES:
        hf_hub_download(
            REPOSITORY,
            name,
            revision=REVISION,
            local_dir=FOLDER,
            token=False,  # Public files: never send a Hugging Face login.
        )
    return FOLDER


def is_downloaded() -> bool:
    """Whether all three files are there, without contacting the server.

    hf_hub_download writes a file into FOLDER only once it is complete, so a
    file that is there is whole.
    """
    return all((FOLDER / name).is_file() for name in FILES)


def read_config(path: Path) -> Config:
    """Read the hyperparameters from config.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Config(
        d=raw["n_embd"],
        n_layers=raw["n_layer"],
        n_heads=raw["n_head"],
        vocab_size=raw["vocab_size"],
        max_tokens=raw["n_positions"],
        eps=raw["layer_norm_epsilon"],
    )


def read_weights(path: Path) -> dict[str, Array]:
    """Read every learned weight in model.safetensors, by name.

    The names come from GPT-2's original code: "wte.weight" is the token
    embedding table, "h.5.attn.c_attn.weight" block 5's W_Q, W_K and W_V.
    Each block also stores "h.N.attn.bias", which is no learned weight
    despite its name but the causal mask, a 1,024 x 1,024 table of ones and
    zeros (model_weights.py shows it). gpt2.py builds the mask itself, so
    these 12 tables are skipped.
    """
    with safe_open(path, framework="numpy") as file:
        return {
            name: file.get_tensor(name)
            for name in file.keys()  # noqa: SIM118 (a file, not a dict)
            if not name.endswith(".attn.bias")
        }


@dataclass(frozen=True)
class Model:
    """GPT-2 small, ready to run: its hyperparameters, weights and tokenizer."""

    config: Config
    weights: dict[str, Array]
    tokenizer: Tokenizer

    @property
    def end_of_text(self) -> int:
        """The ID of <|endoftext|>: 50256, the last entry of the vocabulary."""
        token_id = self.tokenizer.token_to_id(END_OF_TEXT)
        if token_id is None:
            raise ValueError(f"{END_OF_TEXT} is missing from the vocabulary.")
        return token_id

    def encode(self, text: str) -> list[int]:
        """Turn a text into token IDs. GPT-2 adds no start token of its own."""
        return list(self.tokenizer.encode(text).ids)

    def entries(self, token_ids: Sequence[int]) -> list[str]:
        """The vocabulary entry of each ID, as in token_lab: Ġ stands for a space."""
        return [self.tokenizer.id_to_token(i) or f"<{i}>" for i in token_ids]


def read(folder: Path) -> Model:
    """Read the three files from a folder: about half a second."""
    return Model(
        read_config(folder / "config.json"),
        read_weights(folder / "model.safetensors"),
        Tokenizer.from_file(str(folder / "tokenizer.json")),
    )


def load() -> Model:
    """Download the files if needed, then read them."""
    return read(download())
