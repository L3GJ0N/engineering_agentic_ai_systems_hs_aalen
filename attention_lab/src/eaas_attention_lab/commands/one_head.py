"""One attention head by hand: what are S and A for "the cat sat on it"?

Attention is usually explained with the matrices of one head for a short
sentence, with made-up numbers that fit on a slide. This file computes them
with GPT-2 small's real weights: the queries, keys and values, the scores S,
the mask, the softmax, the weights A and what the head returns. Each step is
done by hand, for one head, and checked against the whole model's forward
pass in gpt2.py. Everything is fixed in this file: the text, the layer and
the head.

Setup: uv sync
Run from the project root: uv run one-head
Read next: attention_sink.py
"""

import math

import numpy as np

from eaas_attention_lab import model_files, show
from eaas_attention_lab.gpt2 import causal_mask, forward, layer_norm, softmax

# --- The choices -----------------------------------------------------------------

# The text: GPT-2 reads it as 5 tokens. We put <|endoftext|> in front as a
# start token, because GPT-2's tokenizer adds none. Most heads put much of
# their weight on the first token, whatever it is (attention_sink.py shows
# this), so a word of the sentence in that place would get weight for its
# position alone.
TEXT: str = "the cat sat on it"

# The head: one of 144, head 4 of layer 5, both counted from 0. Try others:
# heads 0 to 11 in layers 0 to 11, and run again.
LAYER: int = 5
HEAD: int = 4

# How many of the 64 columns of Q, K, V and A V to print: enough to see real
# numbers, few enough to fit a terminal line.
SHOWN_COLUMNS: int = 4


def main() -> None:
    """Compute one head step by step and compare with the forward pass."""
    show.utf8_output()
    model = model_files.load()
    config, w = model.config, model.weights
    d, d_h = config.d, config.d_h

    # Step 1: the tokens. Row i of every matrix below belongs to token i.
    ids = [model.end_of_text, *model.encode(TEXT)]
    labels = model.entries(ids)
    t = len(ids)
    print(f"Step 1: {TEXT!r} with <|endoftext|> in front is {t} tokens:")
    print("   i     ID  vocabulary entry")
    for i, (token_id, entry) in enumerate(zip(ids, labels, strict=True)):
        print(f"  {i:>2}  {token_id:>5}  {entry}")

    # The whole model, once: it gives the input of our layer and the results
    # to check our steps against.
    trace = forward(ids, w, config)
    layer = trace.layers[LAYER]

    # Step 2: queries, keys and values of this head. H is the residual stream
    # entering the layer: H1 = X0 + E_pos plus everything blocks 0 to LAYER - 1
    # added (with LAYER = 0, H is H1). Two details that diagrams leave out:
    # attention reads H after LayerNorm, which gives each token's vector mean
    # 0 and variance 1 and then applies learned gains and biases, and each
    # product gets a bias. GPT-2 stores W_Q, W_K and W_V side by side in one
    # matrix, c_attn; we cut it into the three, and take this head's 64
    # columns of each.
    prefix = f"h.{LAYER}."
    x = layer_norm(
        layer.h_in, w[prefix + "ln_1.weight"], w[prefix + "ln_1.bias"], config.eps
    )
    w_q, w_k, w_v = np.split(w[prefix + "attn.c_attn.weight"], 3, axis=1)  # [d, d]
    b_q, b_k, b_v = np.split(w[prefix + "attn.c_attn.bias"], 3)  # [d] each
    head = slice(HEAD * d_h, (HEAD + 1) * d_h)  # this head's columns
    q = x @ w_q[:, head] + b_q[head]  # [T, d_h]
    k = x @ w_k[:, head] + b_k[head]
    v = x @ w_v[:, head] + b_v[head]
    same = all(
        np.allclose(mine, model_value[HEAD], atol=1e-5)
        for mine, model_value in ((q, layer.q), (k, layer.k), (v, layer.v))
    )
    print(
        f"\nStep 2: layer {LAYER}, head {HEAD}: Q = LayerNorm(H) W_Q + b_Q, and "
        f"likewise K and V, {q.shape[0]} x {q.shape[1]} each, one row per token. "
        f"H is the residual stream entering layer {LAYER}. The head uses columns "
        f"{HEAD * d_h} to {(HEAD + 1) * d_h - 1} of W_Q, W_K and W_V. The model "
        f"computed the same: {same}"
    )
    for name, matrix in (("Q", q), ("K", k), ("V", v)):
        print(f"  {name}, the first {SHOWN_COLUMNS} of its {d_h} columns:")
        show.print_matrix(matrix[:, :SHOWN_COLUMNS], labels)

    # Step 3: every query against every key. Row i holds token i's query
    # against the keys of all tokens, including later ones: the mask comes in
    # step 4.
    s = q @ k.T / math.sqrt(d_h)  # [T, T]
    print(f"\nStep 3: the scores S = Q K^T / sqrt({d_h}); row i: token i's query")
    show.print_matrix(s, labels)
    print(
        "  The model computed the same: "
        f"{np.allclose(s, layer.scores[HEAD], atol=1e-5)}"
    )
    i, j = t - 1, t - 2
    print(
        f"  S is not symmetric: S[{i}, {j}] = {s[i, j]:.2f}, but S[{j}, {i}] = "
        f"{s[j, i]:.2f}. The first is {labels[i]}'s query against {labels[j]}'s "
        f"key, the second {labels[j]}'s query against {labels[i]}'s key: queries "
        "and keys come from different matrices."
    )

    # Step 4: the mask, then the softmax. -inf above the diagonal becomes the
    # weight 0, so no token reads a later one. Each row then sums to 1.
    a = softmax(s + causal_mask(t))
    print("\nStep 4: A = softmax(S + mask); row i: how token i spreads its attention")
    show.print_matrix(a, labels, hide_future=True)
    print(
        f"  Each row sums to 1: {np.allclose(a.sum(axis=-1), 1)}. The model computed "
        f"the same: {np.allclose(a, layer.weights[HEAD], atol=1e-5)}"
    )

    # Step 5: what does the division by sqrt(d_h) do? Dividing every score of a
    # row by the same number keeps their order, so the most-read token stays
    # the same. But softmax depends on the differences between the scores:
    # larger differences push the weights towards 0 and 1.
    # Why the division exists: if the 64 numbers of a query and of a key were
    # independent, with mean 0 and variance 1, their dot product would have
    # variance 64; it grows with the width of the head. Dividing by sqrt(64)
    # brings it back to variance 1, for any width. Vaswani et al. suspected
    # that without it, large dot products push the softmax into rows of nearly
    # 0 and 1, where it hardly learns ("Attention Is All You Need", 2017,
    # section 3.2.1). GPT-2 learned W_Q and W_K with the division in place, so
    # the comparison below shows what the division does to these trained
    # scores, not why it is there.
    raw = s * math.sqrt(d_h)  # the scores before the division
    undivided = softmax(raw + causal_mask(t))
    same_order = all(
        np.array_equal(np.argsort(s[r, : r + 1]), np.argsort(raw[r, : r + 1]))
        for r in range(t)
    )
    print(
        f"\nStep 5: without the division by sqrt({d_h}) = {math.sqrt(d_h):g}, the "
        f"order within every row stays the same: {same_order}. The weights "
        f"change, for row {t - 1}, {labels[-1]}:"
    )
    print("  divided:     " + " ".join(f"{value:.2f}" for value in a[-1]))
    print("  not divided: " + " ".join(f"{value:.2f}" for value in undivided[-1]))
    print(
        f"  Why the division is there: if the {d_h} numbers of a query and of a "
        "key were independent, with mean 0 and variance 1, their dot product "
        f"would have variance {d_h}; divided by {math.sqrt(d_h):g}, it has "
        "variance 1, whatever the width. Vaswani et al. suspected that without "
        "the division, large products push the softmax into rows of nearly 0 "
        "and 1, where it hardly learns."
    )

    # Step 6: what if the mask came after the softmax? The weights of the
    # future are then set to 0 after the rows were normalized, so every row but
    # the last no longer sums to 1. What is missing is the share the later
    # tokens took, so each earlier token's result would depend on tokens that
    # come after it.
    late = softmax(s) * ~show.future(t)  # weight 0 where the future is
    print(
        "\nStep 6: masking after the softmax instead leaves rows that sum to "
        + " ".join(f"{value:.2f}" for value in late.sum(axis=-1))
    )
    repaired = late / late.sum(axis=-1, keepdims=True)
    print(
        "  Dividing each row by its sum gives A of step 4 back: "
        f"{np.allclose(repaired, a, atol=1e-6)}. Masking first does that in one go."
    )

    # Step 7: what the head returns. Each token gets a weighted average of the
    # value rows, with its row of A as the weights.
    out = a @ v  # [T, d_h]
    same = np.allclose(out, layer.head_outputs[HEAD], atol=1e-5)
    print(
        f"\nStep 7: the head's result A V is {out.shape[0]} x {out.shape[1]}. The "
        f"model computed the same: {same}"
    )
    print(f"  A V, the first {SHOWN_COLUMNS} of its {d_h} columns:")
    show.print_matrix(out[:, :SHOWN_COLUMNS], labels)
    order = np.argsort(-a[-1])
    print(
        f"  {labels[-1]} gets "
        + " + ".join(f"{a[-1, n]:.2f} x value({labels[n]})" for n in order[:3])
        + " + ..."
    )
    print(
        f"  The {config.n_heads} heads' results, side by side, are {config.n_heads} "
        f"x {d_h} = {d} numbers per token; W_O turns them into the {d} numbers "
        "that attention adds to the residual stream."
    )

    # The figure: S with the future outlined, A, and A with the mask after the
    # softmax, each row's sum beside it.
    fig = show.scores_and_weights(
        s, a, late, labels, f"GPT-2 small, layer {LAYER}, head {HEAD}, for {TEXT!r}"
    )
    show.save(fig, "one_head")


if __name__ == "__main__":
    main()
