"""Step 2: which tokens does Nemotron 3.5 Lightning read for a chat request?

Step 1, tokenize_text.py, encoded a text on its own. But the course's examples
do not send a text: they send a list of messages to a chat API, and the server
writes those messages into one text before the model reads it. This file does
the same with the model's own chat template and shows the resulting tokens.
Everything is fixed in this file: the model, the message and its switches.

Why roles at all? A model only predicts the next token of a sequence and could
continue any text. A chat model, however, was trained further on conversations
written in one fixed format, with markers for whose turn it is. It learned to
answer after the marker that opens the assistant's turn and to end its answer
with the marker that closes a turn; that is how it knows when to stop. The
chat template writes our messages into exactly that format.

It only tokenizes. No model runs and no provider is contacted; the first run
downloads two public files, about 17 MB, from Hugging Face.

Setup: uv sync
Run from the project root: uv run tokenize-chat
Read next: ../../what_you_can_access.md (from the package directory)
"""

import io
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Encoding, Tokenizer

# --- The choice: which model, which message? -----------------------------------

# The same model and revision as in step 1. tokenize_text.py explains how to
# find them on Hugging Face and why the revision is pinned.
REPOSITORY: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
REVISION: str = "bee7596271d1495f6992ae224aefde4410e816b8"

# The message, exactly as a chat completions API receives it: the NVIDIA example
# sends this list as "messages" in its JSON body.
# Try: put "<|im_end|>" into the content and run again. The typed text becomes
# the real marker, token 11: a user could fake the end of their own turn.
MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Explain tokenization in one sentence."}
]

# Downloaded files are kept as plain files in token_lab/downloads, where you can
# open them. The other token_lab commands use the same folder.
FOLDER: Path = (
    Path(__file__).resolve().parents[2]
    / "downloads"
    / "nemotron-3.5-lightning"
    / REVISION[:12]
)


def download(filename: str) -> Path:
    """Fetch one file of the repository at REVISION and return its local path.

    hf_hub_download sends one HTTPS request to
    https://huggingface.co/<REPOSITORY>/resolve/<REVISION>/<filename>
    and keeps the file; later runs reuse it. Downloading the file by hand from
    the website and reading it from disk would work as well.
    """
    path = hf_hub_download(
        REPOSITORY,
        filename,
        revision=REVISION,
        local_dir=FOLDER,
        token=False,  # Public files: never send a Hugging Face login.
    )
    # Only a "dry run", which we never ask for, returns something other than a
    # path. The check is there for the type checker.
    if not isinstance(path, str):
        raise SystemExit(f"Could not download {filename}.")
    return Path(path)


def main() -> None:
    """Show, step by step, how the model turns one chat message into tokens."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows: print pieces like Ġ.
    print(f"Model: {REPOSITORY}, revision {REVISION[:12]}\n")

    # Step 1: the two files that turn chat messages into tokens. The 21.6 GB of
    # weights in the same repository are not needed: this happens before the
    # model runs.
    #   tokenizer.json       the vocabulary, 131,072 entries, and the rules that
    #                        split any text into those entries (step 1)
    #   chat_template.jinja  a small program that writes chat messages into one
    #                        text, in the format the model was trained on
    tokenizer_file: Path = download("tokenizer.json")
    template_file: Path = download("chat_template.jinja")
    print(f"Step 1: tokenizer.json and chat_template.jinja are in {FOLDER}\n")

    # Step 2: one text from the messages. A chat API takes a list of messages,
    # but the model reads a single sequence of tokens. Before encoding, the
    # server runs the model's chat template on the messages; we run it here.
    # The roles become text: "role": "user" turns into <|im_start|>user, a
    # newline, the content and <|im_end|>.
    #
    # The template is written in Jinja, a template language for Python that
    # Hugging Face uses for all chat templates; the jinja2 library runs it.
    #
    # Why a sandbox? The template is code downloaded from the internet. A plain
    # jinja2.Environment would let a template reach Python's internals, e.g.
    # with {{ ''.__class__.__mro__[1].__subclasses__() }}, and from there run
    # commands on your computer. jinja2's SandboxedEnvironment blocks that; the
    # ImmutableSandboxedEnvironment also stops the template from changing the
    # lists and dictionaries we pass in, such as MESSAGES.
    #
    # Why exactly these settings? The Transformers library, which inference
    # servers such as vLLM use for chat templates, runs templates in this
    # environment with these two whitespace settings, so we get the same text
    # a server builds. (For this template the two settings change nothing: it
    # trims its own whitespace with {%- and -%}.)
    #
    # Alternative: transformers.AutoTokenizer does steps 1 to 3 in three lines
    # with apply_chat_template, but hides them and installs a larger library.
    # Both give the same tokens; we compared them on 18 October 2026.
    sandbox: ImmutableSandboxedEnvironment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    template: str = template_file.read_text(encoding="utf-8")
    text: str = sandbox.from_string(template).render(
        messages=MESSAGES,
        # End with the start of the assistant's turn, so that the model's next
        # tokens are its answer. Servers do this for every chat request.
        add_generation_prompt=True,
        # A switch this template defines: no reasoning before the answer. The
        # course's NVIDIA request sends the same value as chat_template_kwargs;
        # the OpenRouter request uses OpenRouter's own reasoning parameter.
        enable_thinking=False,
    )
    print("Step 2: the chat template wrote the message into this text:")
    print(f"  {text!r}\n")

    # Step 3: text to token IDs, as in step 1. The template already wrote every
    # marker, such as <|im_start|>, into the text, and the tokenizer recognizes
    # each one as a token of its own. So we ask it to add nothing extra.
    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_file))
    encoding: Encoding = tokenizer.encode(text, add_special_tokens=False)
    alone: Encoding = tokenizer.encode(MESSAGES[0]["content"], add_special_tokens=False)
    print(
        f"Step 3: the tokenizer split the text into {len(encoding.ids)} tokens; "
        f"the content alone was {len(alone.ids)} in step 1."
    )
    # Are roles special tokens? Only their markers. The tokenizer lists the
    # tokens defined outside its merge rules; the special ones mark structure,
    # not text. The role names, "system", "user" and "ass" + "istant", are
    # ordinary vocabulary entries. <think> and </think> are added tokens too,
    # but not special.
    special_ids: set[int] = {
        token_id
        for token_id, added in tokenizer.get_added_tokens_decoder().items()
        if added.special
    }
    # One row per token: the ID, which is all the model receives, and the
    # vocabulary entry it stands for. In entries, Ġ is a space and Ċ a newline.
    print("      ID  vocabulary entry")
    for token_id, piece in zip(encoding.ids, encoding.tokens, strict=True):
        note: str = "   special token" if token_id in special_ids else ""
        print(f"  {token_id:>6}  {piece}{note}")

    # Step 4: back from IDs to text. The model's answer also arrives as IDs,
    # and decoding turns each chosen ID into text. Decoding our IDs must give
    # the text of step 2 back; we keep the special tokens to compare.
    decoded: str = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    print(
        f"\nStep 4: decoding the {len(encoding.ids)} IDs gives back the text "
        f"of step 2: {decoded == text}"
    )


if __name__ == "__main__":
    main()
