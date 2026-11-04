"""GPT-2 small in NumPy: the whole forward pass, keeping the attention matrices.

From token IDs to the scores for the next token: one row of the embedding
table per token, plus a position vector; then 12 blocks, each with attention
and a feed-forward network; then the output layer. forward() keeps the
queries, keys, values, scores and weights of every head and the residual
stream after every step in a Trace, so that the scripts can print and plot
GPT-2's real numbers instead of made-up ones. Hugging Face Transformers hands
out only part of this, and the queries and the scores before the softmax not
at all.

GPT-2 does two things that attention diagrams usually leave out, and so does
this file:
  - Residual connections. Each token's vector is a running sum, called the
    residual stream. Attention and the feed-forward network each read it and
    ADD their result to it, instead of replacing it.
  - LayerNorm. What attention and the feed-forward network read is a copy of
    the residual stream with each token's vector shifted and scaled to mean 0
    and variance 1, then multiplied by learned gains and shifted by learned
    biases. The final output gets the same treatment once more.

Shapes are given as [rows, columns]: T is the number of tokens, d = 768 the
width of a token's vector, d_h = 64 the width of one head. Layers and heads
count from 0, as everything in Python does: layer 5 is the sixth block.
All arrays are float32, the precision the weights are stored in.
"""

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# An array of floating-point numbers. At run time they are float32, which
# tests/test_gpt2.py checks; NumPy's type hints cannot follow float32 through
# every operation, so the type checker only knows "floating point".
Array: TypeAlias = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class Config:
    """GPT-2's hyperparameters, read from its config.json."""

    d: int  # width of a token's vector, "n_embd": 768
    n_layers: int  # number of blocks, "n_layer": 12
    n_heads: int  # heads per block, "n_head": 12
    vocab_size: int  # entries in the vocabulary, "vocab_size": 50,257
    max_tokens: int  # longest input, "n_positions": 1,024 tokens
    eps: float  # keeps LayerNorm from dividing by 0, "layer_norm_epsilon"

    @property
    def d_h(self) -> int:
        """The width of one head: 768 / 12 = 64."""
        return self.d // self.n_heads


@dataclass(frozen=True)
class LayerTrace:
    """Everything one block computes for an input of T tokens."""

    h_in: Array  # [T, d] residual stream entering the block
    q: Array  # [heads, T, d_h] queries, one matrix per head
    k: Array  # [heads, T, d_h] keys
    v: Array  # [heads, T, d_h] values
    scores: Array  # [heads, T, T] S = Q K^T / sqrt(d_h), before the mask
    weights: Array  # [heads, T, T] A = softmax(S + mask), one row per token
    head_outputs: Array  # [heads, T, d_h] A V, per head
    attn_out: Array  # [T, d] concat(heads) W_O + b_O: what attention adds
    h_mid: Array  # [T, d] h_in + attn_out
    h_out: Array  # [T, d] h_mid + what the feed-forward network adds


@dataclass(frozen=True)
class Trace:
    """The forward pass of one input, from token IDs to the logits."""

    token_ids: list[int]
    x0: Array  # [T, d] one row of the token embedding table E per token
    h1: Array  # [T, d] X0 + E_pos: the input of the first block
    layers: list[LayerTrace]
    final: Array  # [T, d] the last block's output after the final LayerNorm
    logits: Array  # [T, V] one score per vocabulary entry, for the next token


def layer_norm(x: Array, gain: Array, bias: Array, eps: float) -> Array:
    """Give each row mean 0 and variance 1, then apply the learned gain and bias."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(variance + eps) + bias


def gelu(x: Array) -> Array:
    """GPT-2's activation function: GELU, in the tanh form GPT-2 was trained with."""
    # math.sqrt, not np.sqrt: NumPy's float64 would turn the result into
    # float64, while a Python float keeps the array's float32.
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def softmax(scores: Array) -> Array:
    """Turn each row into weights that are positive and sum to 1."""
    # Subtracting the row's maximum changes no weight but keeps exp() from
    # overflowing. exp(-inf) = 0: a masked entry gets the weight 0.
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def causal_mask(t: int) -> Array:
    """[T, T]: 0 where token i may read token j (j <= i), -inf above the diagonal."""
    return np.triu(np.full((t, t), -np.inf, dtype=np.float32), k=1)


def split_heads(x: Array, n_heads: int) -> Array:
    """[T, d] to [heads, T, d_h]: head j gets columns j*d_h to (j+1)*d_h - 1."""
    t, d = x.shape
    return x.reshape(t, n_heads, d // n_heads).transpose(1, 0, 2)


def merge_heads(x: Array) -> Array:
    """[heads, T, d_h] to [T, d]: put the heads' results side by side again."""
    n_heads, t, d_h = x.shape
    return x.transpose(1, 0, 2).reshape(t, n_heads * d_h)


def block(h: Array, weights: dict[str, Array], i: int, config: Config) -> LayerTrace:
    """Block i: attention, then the feed-forward network, each with a residual."""

    def w(name: str) -> Array:
        return weights[f"h.{i}.{name}"]

    # --- Attention: each token reads the tokens up to itself ------------------
    x = layer_norm(h, w("ln_1.weight"), w("ln_1.bias"), config.eps)
    # GPT-2 stores W_Q, W_K and W_V side by side, as one [768, 2304] matrix
    # "c_attn" with one bias, so one product computes all three. It stores
    # weights as [inputs, outputs], so x @ W needs no transpose.
    qkv = x @ w("attn.c_attn.weight") + w("attn.c_attn.bias")  # [T, 3d]
    q_all, k_all, v_all = np.split(qkv, 3, axis=-1)  # each [T, d]
    q = split_heads(q_all, config.n_heads)  # [heads, T, d_h]
    k = split_heads(k_all, config.n_heads)
    v = split_heads(v_all, config.n_heads)

    # One score per pair of tokens and head: how well token i's query matches
    # token j's key. The mask then hides the future, and softmax turns each
    # row into weights.
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(config.d_h)  # [heads, T, T]
    attention_weights = softmax(scores + causal_mask(len(h)))
    head_outputs = attention_weights @ v  # [heads, T, d_h]: weighted averages

    # W_O ("c_proj") turns the heads' results, side by side, into one vector
    # per token, which is added to the residual stream.
    concat = merge_heads(head_outputs)  # [T, d]
    attn_out = concat @ w("attn.c_proj.weight") + w("attn.c_proj.bias")
    h_mid = h + attn_out

    # --- Feed-forward network: each token on its own -------------------------
    x = layer_norm(h_mid, w("ln_2.weight"), w("ln_2.bias"), config.eps)
    hidden = gelu(x @ w("mlp.c_fc.weight") + w("mlp.c_fc.bias"))  # [T, 4d]
    h_out = h_mid + hidden @ w("mlp.c_proj.weight") + w("mlp.c_proj.bias")

    return LayerTrace(
        h_in=h,
        q=q,
        k=k,
        v=v,
        scores=scores,
        weights=attention_weights,
        head_outputs=head_outputs,
        attn_out=attn_out,
        h_mid=h_mid,
        h_out=h_out,
    )


def forward(token_ids: list[int], weights: dict[str, Array], config: Config) -> Trace:
    """Run GPT-2 on a list of token IDs, keeping every head's matrices."""
    t = len(token_ids)
    if not 0 < t <= config.max_tokens:
        raise ValueError(f"GPT-2 reads 1 to {config.max_tokens} tokens, not {t}.")

    x0 = weights["wte.weight"][token_ids]  # [T, d]: row token_id of E, per token
    h1 = x0 + weights["wpe.weight"][:t]  # [T, d]: + rows 0 to T-1 of E_pos

    layers: list[LayerTrace] = []
    h: Array = h1
    for i in range(config.n_layers):
        layers.append(block(h, weights, i, config))
        h = layers[-1].h_out

    final = layer_norm(h, weights["ln_f.weight"], weights["ln_f.bias"], config.eps)
    # The output layer has no matrix of its own: GPT-2 reuses E ("tied"
    # weights), so the logit of vocabulary entry j is row . E[j].
    logits = final @ weights["wte.weight"].T  # [T, V]
    return Trace(list(token_ids), x0, h1, layers, final, logits)
