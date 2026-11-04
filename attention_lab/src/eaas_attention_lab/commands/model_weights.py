"""Where are W_Q, W_K, W_V and W_O in a real model file, and what else is in it?

Attention diagrams show four weight matrices per block: W_Q, W_K and W_V turn
each token's vector into a query, a key and a value, and W_O turns the heads'
results back into one vector per token. This file opens the weights of GPT-2
small and finds them: their names, shapes and sizes. It also lists what else
the file holds that diagrams rarely show: biases, LayerNorm, the feed-forward
networks and a stored causal mask, and the one thing it lacks, a matrix for
the output layer.

The first run downloads three public files, 550 MB, from Hugging Face; run it
once on a fast connection, for example at home. Later runs reuse the files
and work offline. No account is needed.

Setup: uv sync
Run from the project root: uv run model-weights
Read next: one_head.py
"""

import struct

import numpy as np
from safetensors import safe_open

from eaas_attention_lab import model_files, show

# What each tensor of a block holds, by the end of its name.
BLOCK_TENSORS: dict[str, str] = {
    "ln_1.weight": "LayerNorm before attention: gains",
    "ln_1.bias": "LayerNorm before attention: biases",
    "attn.c_attn.weight": "W_Q, W_K and W_V, side by side",
    "attn.c_attn.bias": "their biases",
    "attn.c_proj.weight": "W_O",
    "attn.c_proj.bias": "its bias",
    "attn.bias": "not a bias: the causal mask (step 5)",
    "ln_2.weight": "LayerNorm before the feed-forward network: gains",
    "ln_2.bias": "LayerNorm before the feed-forward network: biases",
    "mlp.c_fc.weight": "feed-forward network, first layer: d to 4d",
    "mlp.c_fc.bias": "its bias",
    "mlp.c_proj.weight": "feed-forward network, second layer: 4d to d",
    "mlp.c_proj.bias": "its bias",
}
OTHER_TENSORS: dict[str, str] = {
    "wte.weight": "token embedding table E: one row per vocabulary entry",
    "wpe.weight": "position embedding table E_pos: one row per position",
    "ln_f.weight": "final LayerNorm: gains",
    "ln_f.bias": "final LayerNorm: biases",
}


def main() -> None:
    """List the tensors of the model file, find the attention matrices, count."""
    show.utf8_output()
    print(f"Model: {model_files.REPOSITORY}, revision {model_files.REVISION[:12]}\n")

    # Step 1: the files. download() fetches only what is not there yet.
    folder = model_files.download()
    print(f"Step 1: three files in {folder}")
    for name in model_files.FILES:
        print(f"  {name:<18} {(folder / name).stat().st_size:>13,} bytes")

    # Step 2: the hyperparameters, a small JSON file. Open it to see them all;
    # it uses GPT-2's own names, given here with the diagrams' names.
    config = model_files.read_config(folder / "config.json")
    d, d_h = config.d, config.d_h
    print(
        f"\nStep 2: config.json: n_embd = {d}, the width d of a token's vector; "
        f"n_layer = {config.n_layers} blocks; n_head = {config.n_heads} heads, so "
        f"d_h = {d} / {config.n_heads} = {d_h} (computed, not in the file); "
        f"vocab_size = {config.vocab_size:,} entries; n_positions = "
        f"{config.max_tokens:,} tokens at most"
    )

    # Step 3: the weights. A safetensors file is a small JSON header, naming
    # every array with its shape and data type, followed by the raw numbers.
    path = folder / "model.safetensors"
    with safe_open(path, framework="numpy") as file:
        names: list[str] = list(file.keys())
        shapes = {name: list(file.get_slice(name).get_shape()) for name in names}
        dtypes = {file.get_slice(name).get_dtype() for name in names}
        masks = [file.get_tensor(f"h.{i}.attn.bias") for i in range(config.n_layers)]
    print(
        f"\nStep 3: model.safetensors holds {len(names)} named arrays (tensors), "
        f"data type {', '.join(sorted(dtypes))}: 32-bit floats"
    )
    print(f"  {len(OTHER_TENSORS)} outside the blocks:")
    for name, meaning in OTHER_TENSORS.items():
        print(f"    {name:<24} {shapes[name]!s:<20} {meaning}")
    print(f"  {len(BLOCK_TENSORS)} in each of the {config.n_layers} blocks; block 0:")
    for suffix, meaning in BLOCK_TENSORS.items():
        name = f"h.0.{suffix}"
        print(f"    {name:<24} {shapes[name]!s:<20} {meaning}")
    expected = len(OTHER_TENSORS) + config.n_layers * len(BLOCK_TENSORS)
    print(
        f"  {len(OTHER_TENSORS)} + 12 x {len(BLOCK_TENSORS)} = {expected}: "
        f"{expected == len(names)}"
    )

    # Step 4: the four matrices of the diagrams. GPT-2 keeps W_Q, W_K and W_V
    # in one matrix, so that one product computes queries, keys and values.
    # Weights are stored as [inputs, outputs]: a token's vector x, 768
    # numbers, times the matrix gives the result, x @ W, without transposing.
    print("\nStep 4: the attention matrices of a block, here block 0")
    for part, matrix in enumerate(("W_Q", "W_K", "W_V")):
        first, last = part * d, (part + 1) * d - 1
        print(
            f"  {matrix} = columns {first:>4} to {last:>4} of h.0.attn.c_attn.weight: "
            f"[{d}, {d}]"
        )
    print(f"  W_O = h.0.attn.c_proj.weight: {shapes['h.0.attn.c_proj.weight']}")
    print(
        f"  Each head uses {d_h} of the {d} columns of W_Q, W_K and W_V: head 0 "
        f"columns 0 to {d_h - 1}, head 1 {d_h} to {2 * d_h - 1}, ..., head 11 "
        f"{11 * d_h} to {12 * d_h - 1}. W_O mixes all heads' results."
    )
    attention = 4 * d * d
    attention_biases = shapes["h.0.attn.c_attn.bias"][0] + d
    print(
        f"  4 x {d} x {d} = {attention:,} weights in these four matrices of every "
        f"block, plus {attention_biases:,} biases (step 5)"
    )

    # Step 5: what the diagrams leave out.
    print("\nStep 5: in the file, but rarely in diagrams")
    print(
        "  - Biases: every matrix in a block has one, added after the product, such "
        f"as the {shapes['h.0.attn.c_attn.bias'][0]:,} numbers of c_attn.bias for "
        f"W_Q, W_K and W_V and the {d} of c_proj.bias for W_O."
    )
    print(
        "  - LayerNorm: before attention, before the feed-forward network and at the "
        f"end, {d} gains and {d} biases each."
    )
    ffn = shapes["h.0.mlp.c_fc.weight"][1] * d * 2
    print(
        f"  - The feed-forward network: [{d}, {4 * d}] and [{4 * d}, {d}], "
        f"8 x d^2 = {ffn:,} weights per block, twice as many as attention."
    )
    # "attn.bias" is not learned. Hugging Face's first PyTorch version of
    # GPT-2 (2019) kept the causal mask as a table inside every attention
    # layer, so it was saved together with the weights. That code used it to
    # push the scores above the diagonal to a huge negative number before the
    # softmax: the same effect as the -inf in gpt2.py. The Transformers
    # library has ignored the stored tables since 2023, and gpt2.py skips them
    # too.
    lower = np.tril(np.ones((config.max_tokens, config.max_tokens), np.float32))
    mask_ok = all(np.array_equal(mask[0, 0], lower) for mask in masks)
    print(
        f"  - h.N.attn.bias is the causal mask: {config.max_tokens:,} x "
        f"{config.max_tokens:,} numbers, 1 on and below the diagonal, 0 above it, "
        f"the same in all 12 blocks: {mask_ok}. Not learned: gpt2.py skips it."
    )

    # Step 6: what is missing. The output layer turns the final vector of a
    # token into one score per vocabulary entry; GPT-2 reuses E for it.
    lm_head = [name for name in names if name.startswith("lm_head")]
    print(
        f"\nStep 6: no matrix for the output layer: {lm_head == []}. GPT-2 reuses the "
        f"embedding table E (tied weights): the score of entry j is the final "
        f"vector times row j of E. That saves {config.vocab_size * d:,} weights."
    )

    # Step 7: where the numbers are. The masks are no weights; we count them
    # separately, to check that the file holds nothing else.
    sizes = {name: int(np.prod(shape)) for name, shape in shapes.items()}
    groups = {
        "token embeddings E": sizes["wte.weight"],
        "position embeddings E_pos": sizes["wpe.weight"],
        "attention, 12 blocks": sum(
            n
            for name, n in sizes.items()
            if ".attn.c_" in name  # c_attn, c_proj
        ),
        "feed-forward networks, 12 blocks": sum(
            n for name, n in sizes.items() if ".mlp." in name
        ),
        "LayerNorms, 25": sum(
            n
            for name, n in sizes.items()
            if "ln_" in name  # ln_1, ln_2, ln_f
        ),
    }
    weights = sum(groups.values())
    mask_numbers = sum(n for name, n in sizes.items() if name.endswith(".attn.bias"))
    print(f"\nStep 7: {weights:,} learned weights, the masks not counted:")
    for group, n in groups.items():
        print(f"  {group:<34} {n:>12,}  {n / weights:7.2%}")
    # The file starts with 8 bytes that give the header's length.
    with path.open("rb") as raw:
        (header,) = struct.unpack("<Q", raw.read(8))
    size = 4 * (weights + mask_numbers) + 8 + header
    print(
        f"  At 4 bytes per number: {4 * weights:,} bytes of weights, "
        f"{4 * mask_numbers:,} of masks and {8 + header:,} of header make "
        f"{size:,} bytes, the size of the file: {size == path.stat().st_size}"
    )

    # The figure: one bar per group, with its size and share.
    notes = [
        f"{n / 1e6:.1f} M ({n / weights:.2%})"
        if n >= 1e6
        else f"{n:,} ({n / weights:.2%})"
        for n in groups.values()
    ]
    fig = show.labelled_bars(
        groups,
        notes,
        f"Where GPT-2 small's {weights / 1e6:.0f} million weights are",
        "weights",
        unit=(1e6, " M"),
    )
    show.save(fig, "model_weights")


if __name__ == "__main__":
    main()
