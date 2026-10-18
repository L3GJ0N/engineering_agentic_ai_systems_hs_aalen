"""Step 1: which tokens does a text become for Nemotron 3.5 Lightning?

The NVIDIA example in llm_inference sends the text "Explain tokenization in
one sentence." in its JSON body. A model never reads text, only token IDs. This
file shows the tokens that text becomes with the tokenizer of the model
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, and turns them back into text.
Everything is fixed in this file: the model and the text.

It only tokenizes. No model runs and no provider is contacted; the first run
downloads one public file, about 17 MB, from Hugging Face.
Step 2, tokenize_chat.py, shows what the server adds around the text before
the model reads it.

Setup: uv sync
Run from the project root: uv run tokenize-text
"""

import io
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Encoding, Tokenizer

# --- The choice: which model, which text? --------------------------------------

# Every model comes with its own tokenizer, fixed before its training: the same
# text gives different tokens for different models. So we first choose a model.
#
# Open-weight models are published on Hugging Face in repositories, much like
# code on GitHub. This repository holds the model that the course's NVIDIA and
# OpenRouter examples call:
# https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
#
# How to find it. The provider's model ID, nvidia/nemotron-3.5-lightning-30b-a3b
# at NVIDIA Build, is not a Hugging Face name; two routes lead to the
# repository:
#   - From the provider: NVIDIA Build's page for the model links to it as
#     "Hugging Face" under "Release Date".
#   - By search: type nemotron-3.5-lightning into the search box on
#     huggingface.co. Many results are copies by other users; take the one
#     owned by the publisher, here nvidia.
# The name is owner/repository, as in the page heading and in the address
# after huggingface.co/. Its tab "Files and versions" must list tokenizer.json.
# README.md, "Find a model's tokenizer on Hugging Face", has the steps in
# detail, including which variant to choose.
REPOSITORY: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

# Why a revision? A repository has a history, like a Git repository: its owner
# can upload new files at any time, and the branch "main" then points to them.
# A revision names one version. We pin the full commit hash of the version
# checked on 18 October 2026, so the file, and with it the tokens, never
# change under us, and everyone in the course sees the same result.
# To get the hash: on the tab "Files and versions", open "History: ... commits"
# and click a commit; the address then ends with its full hash. Or open
# https://huggingface.co/api/models/<REPOSITORY> and copy the value of "sha",
# the newest commit.
# REVISION = "main" would work too and always fetch the newest files, at the
# price of results that can change without notice.
REVISION: str = "bee7596271d1495f6992ae224aefde4410e816b8"

# The text: the "content" the NVIDIA example sends in its JSON body.
TEXT: str = "Explain tokenization in one sentence."

# Downloaded files are kept as plain files in token_lab/downloads, where you can
# open them. The other token_lab commands use the same folder.
FOLDER: Path = (
    Path(__file__).resolve().parents[2]
    / "downloads"
    / "nemotron-3.5-lightning"
    / REVISION[:12]
)


def main() -> None:
    """Encode one text with the model's tokenizer, print the tokens, decode."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows: print pieces like Ġ.
    print(f"Model: {REPOSITORY}, revision {REVISION[:12]}\n")

    # Step 1: the tokenizer file. Of the repository's files, only tokenizer.json
    # is needed to turn text into tokens; the 21.6 GB of weights are not, since
    # tokenization happens before the model runs. tokenizer.json holds the
    # vocabulary, 131,072 entries, and the rules that split any text into them.
    #
    # hf_hub_download sends one HTTPS request to
    # https://huggingface.co/<REPOSITORY>/resolve/<REVISION>/tokenizer.json
    # and keeps the file; later runs reuse it. Downloading the file by hand from
    # the website and reading it from disk would work as well.
    path = hf_hub_download(
        REPOSITORY,
        "tokenizer.json",
        revision=REVISION,
        local_dir=FOLDER,
        token=False,  # A public file: never send a Hugging Face login.
    )
    # Only a "dry run", which we never ask for, returns something other than a
    # path. The check is there for the type checker.
    if not isinstance(path, str):
        raise SystemExit("Could not download tokenizer.json.")
    print(f"Step 1: tokenizer.json is in {FOLDER}\n")

    # Step 2: text to token IDs. Hugging Face's tokenizers library reads
    # tokenizer.json; the larger Transformers library uses it underneath.
    # encode() splits the text into vocabulary entries and returns their IDs.
    # Some tokenizers also add a start token, such as <s>, to every text;
    # Nemotron's adds none.
    tokenizer: Tokenizer = Tokenizer.from_file(path)
    encoding: Encoding = tokenizer.encode(TEXT)
    print(f"Step 2: the text {TEXT!r} is {len(encoding.ids)} tokens:")
    # One row per token: the ID, which is all the model receives, and the
    # vocabulary entry it stands for. Ġ marks a space at the start of an entry.
    print("      ID  vocabulary entry")
    for token_id, piece in zip(encoding.ids, encoding.tokens, strict=True):
        print(f"  {token_id:>6}  {piece}")

    # Step 3: back from IDs to text. The model's answer also arrives as IDs,
    # and decoding turns each chosen ID into text.
    decoded: str = tokenizer.decode(encoding.ids)
    print(f"\nStep 3: decoding the IDs gives back the text: {decoded == TEXT}")
    print(
        "\nThese are the tokens of the text alone. Before the model reads it, "
        "the server writes it into a chat format with more tokens: see step 2, "
        "uv run tokenize-chat."
    )


if __name__ == "__main__":
    main()
