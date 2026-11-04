"""gpt2.py must compute what Hugging Face Transformers computes.

The reference is Transformers' GPT-2 on PyTorch, with its "eager" attention:
the plain formulas, the only variant that returns the attention weights.
Both read the same downloaded files.

Optional: it needs the dependency group "reference", PyTorch and
Transformers, 400 to 850 MB that no script needs. Run it with
    uv run --group reference pytest
Without that group, or before the GPT-2 files are downloaded, it is skipped.
"""

from typing import Any

import numpy as np
import pytest

from eaas_attention_lab import model_files
from eaas_attention_lab.gpt2 import Array, forward

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

FOLDER = model_files.FOLDER
pytestmark = pytest.mark.skipif(
    not all((FOLDER / name).is_file() for name in model_files.FILES),
    reason="GPT-2 files not downloaded yet: run uv run model-weights",
)

TEXTS: tuple[str, ...] = (
    "the cat sat on it",
    "Attention mixes information between positions; the feed-forward network "
    "works on each position separately. Both add to the residual stream.",
)


@pytest.fixture(scope="module")
def model() -> model_files.Model:
    return model_files.read(FOLDER)


@pytest.fixture(scope="module")
def reference() -> Any:
    gpt2 = transformers.GPT2LMHeadModel.from_pretrained(
        FOLDER, attn_implementation="eager", local_files_only=True
    )
    return gpt2.eval()


def largest_difference(ours: Array, theirs: Any) -> float:
    return float(np.max(np.abs(ours - theirs.numpy())))


@pytest.mark.parametrize("text", TEXTS, ids=("short", "long"))
def test_weights_hidden_states_and_logits_agree(
    model: model_files.Model, reference: Any, text: str
) -> None:
    ids = [model.end_of_text, *model.encode(text)]
    ours = forward(ids, model.weights, model.config)
    with torch.no_grad():
        theirs = reference(
            torch.tensor([ids]), output_attentions=True, output_hidden_states=True
        )
    assert reference.dtype == torch.float32

    for layer, weights in zip(ours.layers, theirs.attentions, strict=True):
        assert largest_difference(layer.weights, weights[0]) < 1e-4

    # Transformers' 13 hidden states: H1, the outputs of blocks 0 to 10, and
    # the output of block 11 after the final LayerNorm.
    hidden = [ours.h1, *(layer.h_out for layer in ours.layers[:-1]), ours.final]
    for mine, other in zip(hidden, theirs.hidden_states, strict=True):
        scale = float(np.max(np.abs(mine)))
        assert largest_difference(mine, other[0]) < 1e-5 * scale

    logits = theirs.logits[0]
    assert largest_difference(ours.logits, logits) < 1e-2
    assert np.array_equal(ours.logits.argmax(axis=-1), logits.numpy().argmax(axis=-1))
