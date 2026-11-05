"""The forward pass on a tiny random GPT-2, without any download.

The weights are random numbers with GPT-2's names and layout, 12 blocks with
12 heads, but heads of 4 numbers (d = 48) and a vocabulary of 50 (gpt2.TINY),
so these tests run anywhere in a moment. They check what the scripts rely on,
not GPT-2's numbers: test_scripts.py and test_reference.py check those.
"""

import numpy as np
import pytest

from eaas_attention_lab.gpt2 import (
    TINY,
    Array,
    causal_mask,
    forward,
    merge_heads,
    random_weights,
    softmax,
    split_heads,
)

CONFIG = TINY
IDS = [3, 14, 15, 9, 26, 5, 35]


@pytest.fixture(scope="module")
def weights() -> dict[str, Array]:
    return random_weights(CONFIG, seed=0)


def test_every_matrix_has_its_shape_and_stays_float32(
    weights: dict[str, Array],
) -> None:
    trace = forward(IDS, weights, CONFIG)
    t, layer = len(IDS), trace.layers[0]
    assert layer.q.shape == (CONFIG.n_heads, t, CONFIG.d_h)
    assert layer.weights.shape == (CONFIG.n_heads, t, t)
    assert trace.logits.shape == (t, CONFIG.vocab_size)
    # The stacked views put the layer first: attention[layer, head, i, j].
    assert trace.attention.shape == (CONFIG.n_layers, CONFIG.n_heads, t, t)
    np.testing.assert_array_equal(trace.attention[1, 2], trace.layers[1].weights[2])
    assert trace.block_outputs.shape == (CONFIG.n_layers, t, CONFIG.d)
    # NumPy turns float32 into float64 when a NumPy float64 joins in; the
    # forward pass must stay in the weights' precision.
    assert layer.weights.dtype == np.float32
    assert trace.logits.dtype == np.float32


def test_each_row_of_a_sums_to_1_and_puts_nothing_on_the_future(
    weights: dict[str, Array],
) -> None:
    future = np.triu(np.ones((len(IDS), len(IDS)), dtype=bool), k=1)
    for layer in forward(IDS, weights, CONFIG).layers:
        np.testing.assert_allclose(layer.weights.sum(axis=-1), 1, rtol=1e-6)
        assert np.all(layer.weights[:, future] == 0)


def test_one_more_token_changes_no_earlier_row(weights: dict[str, Array]) -> None:
    # What makes a KV cache possible: the keys and values of earlier tokens
    # stay the same when a token is appended.
    short = forward(IDS[:-1], weights, CONFIG)
    long = forward(IDS, weights, CONFIG)
    t = len(IDS) - 1
    for before, after in zip(short.layers, long.layers, strict=True):
        np.testing.assert_allclose(after.k[:, :t], before.k, atol=1e-6)
        np.testing.assert_allclose(after.v[:, :t], before.v, atol=1e-6)
    np.testing.assert_allclose(long.logits[:t], short.logits, atol=1e-5)


def test_the_heads_side_by_side_times_w_o_is_what_attention_adds(
    weights: dict[str, Array],
) -> None:
    layer = forward(IDS, weights, CONFIG).layers[1]
    added = (
        merge_heads(layer.head_outputs) @ weights["h.1.attn.c_proj.weight"]
        + weights["h.1.attn.c_proj.bias"]
    )
    np.testing.assert_allclose(added, layer.h_mid - layer.h_in, atol=1e-5)


def test_head_j_gets_its_own_columns_and_merging_undoes_splitting() -> None:
    x = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
    heads = split_heads(x, 4)
    assert heads.shape == (4, 5, 4)
    np.testing.assert_array_equal(heads[1], x[:, 4:8])
    np.testing.assert_array_equal(merge_heads(heads), x)


def test_softmax_gives_masked_entries_0_and_survives_large_scores() -> None:
    # The largest score sits in the masked cell: without the mask, row 0 would
    # put its weight there; without subtracting the maximum, exp() overflows.
    scores = np.array([[1000.0, 2000.0], [5.0, 5.0]], dtype=np.float32)
    expected = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(softmax(scores + causal_mask(2)), expected)


def test_more_tokens_than_positions_are_refused(weights: dict[str, Array]) -> None:
    with pytest.raises(ValueError, match="1 to 32 tokens"):
        forward(list(range(33)), weights, CONFIG)
