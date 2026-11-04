"""The forward pass on a tiny random GPT-2, without any download.

The weights are random numbers with GPT-2's names and layout, but d = 16,
4 heads of 4, 2 blocks and a vocabulary of 50, so these tests run anywhere in
a moment. They check what the scripts rely on, not GPT-2's numbers:
test_scripts.py and test_reference.py check those.
"""

import numpy as np
import pytest

from eaas_attention_lab.gpt2 import (
    Array,
    Config,
    causal_mask,
    forward,
    merge_heads,
    softmax,
    split_heads,
)

CONFIG = Config(d=16, n_layers=2, n_heads=4, vocab_size=50, max_tokens=32, eps=1e-5)
IDS = [3, 14, 15, 9, 26, 5, 35]


@pytest.fixture(scope="module")
def weights() -> dict[str, Array]:
    rng = np.random.default_rng(0)
    d = CONFIG.d

    def random(*shape: int) -> Array:
        return (0.3 * rng.standard_normal(shape)).astype(np.float32)

    w = {
        "wte.weight": random(CONFIG.vocab_size, d),
        "wpe.weight": random(CONFIG.max_tokens, d),
        "ln_f.weight": 1 + random(d),
        "ln_f.bias": random(d),
    }
    for i in range(CONFIG.n_layers):
        w |= {
            f"h.{i}.ln_1.weight": 1 + random(d),
            f"h.{i}.ln_1.bias": random(d),
            f"h.{i}.attn.c_attn.weight": random(d, 3 * d),
            f"h.{i}.attn.c_attn.bias": random(3 * d),
            f"h.{i}.attn.c_proj.weight": random(d, d),
            f"h.{i}.attn.c_proj.bias": random(d),
            f"h.{i}.ln_2.weight": 1 + random(d),
            f"h.{i}.ln_2.bias": random(d),
            f"h.{i}.mlp.c_fc.weight": random(d, 4 * d),
            f"h.{i}.mlp.c_fc.bias": random(4 * d),
            f"h.{i}.mlp.c_proj.weight": random(4 * d, d),
            f"h.{i}.mlp.c_proj.bias": random(d),
        }
    return w


def test_every_matrix_has_its_shape_and_stays_float32(
    weights: dict[str, Array],
) -> None:
    trace = forward(IDS, weights, CONFIG)
    t, layer = len(IDS), trace.layers[0]
    assert layer.q.shape == (CONFIG.n_heads, t, CONFIG.d_h)
    assert layer.weights.shape == (CONFIG.n_heads, t, t)
    assert trace.logits.shape == (t, CONFIG.vocab_size)
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
