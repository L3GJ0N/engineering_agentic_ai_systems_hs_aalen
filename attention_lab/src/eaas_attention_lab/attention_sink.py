"""Why do most heads put their weight on the first token?

In one_head.py, the early tokens put most of their attention weight on the
first token, <|endoftext|>, a token without any meaning in the sentence. This
file measures that over all 144 heads of GPT-2 small and asks three
questions: does it depend on the word or on the position? What does a head
get from there? And how does the first position stand out inside the model?
The answers matter whenever you read an attention map, and for servers that
shorten the memory of a long conversation.

Everything is fixed in this file: the texts, and the layer of the figure.

Setup: uv sync
Run from the project root: uv run attention-sink
"""

import numpy as np

from eaas_attention_lab import model_files, show
from eaas_attention_lab.gpt2 import Array, Trace, forward

# --- The choices -----------------------------------------------------------------

# The text, by default that of one_head.py. Step 2 reads it with <|endoftext|>
# in front, then without it, then behind each of these first words.
TEXT: str = "the cat sat on it"
FIRST_WORDS: tuple[str, ...] = ("Yesterday", "So")
# The same token, Ġthe, at positions 0 and 4: the leading space makes the
# first "the" the same vocabulary entry as the second.
REPEATED: str = " the cat sat on the mat"

# Step 3 compares the heads that put more than PARKED of the last token's
# weight on the first token with those that put less than READING there.
PARKED: float = 0.9
READING: float = 0.2

# The layer whose 12 heads the figure shows. Try others: 0 to 11.
LAYER: int = 5


def last_rows(trace: Trace) -> Array:
    """[layers, heads, T]: the last token's attention weights in every head."""
    return np.stack([layer.weights[:, -1, :] for layer in trace.layers])


def on_first(trace: Trace) -> tuple[int, float]:
    """Count the heads whose largest weight is on the first token; mean weight.

    Both for the last token: in how many heads its largest weight is on the
    first token, and the mean weight it puts there.
    """
    rows = last_rows(trace)
    return int((rows.argmax(axis=-1) == 0).sum()), float(rows[..., 0].mean())


def result_shares(trace: Trace) -> Array:
    """[layers, heads]: the first token's share of what each head returns.

    A head returns, for the last token, the sum of weight x value over all
    tokens. The share is the length of the first token's part divided by the
    sum of the lengths of all parts.
    """
    shares = []
    for layer in trace.layers:
        parts = layer.weights[:, -1, :, np.newaxis] * layer.v  # [heads, T, d_h]
        lengths = np.linalg.norm(parts, axis=-1)  # [heads, T]
        shares.append(lengths[:, 0] / lengths.sum(axis=-1))
    return np.stack(shares)


def main() -> None:
    """Measure where the weight goes, what it brings and what marks position 0."""
    show.utf8_output()
    model = model_files.load()
    heads = model.config.n_layers * model.config.n_heads

    # Step 1: the last token of the sentence, " it", in all 144 heads.
    ids = [model.end_of_text, *model.encode(TEXT)]
    trace = forward(ids, model.weights, model.config)
    count, mean = on_first(trace)
    print(
        f"Step 1: <|endoftext|> + {TEXT!r}: in {count} of {heads} heads, "
        f"{model.entries(ids)[-1]} puts its largest weight on the first token, "
        f"<|endoftext|>; on average {mean:.2f} of its weight."
    )

    # Step 2: change the first token. If the weight followed the word, it would
    # move with it; it stays on position 0.
    print("\nStep 2: the word or the position?")
    print(f"  {model.entries(ids)[-1]}'s weight on the first token, by first token:")
    print("  first token    heads with the largest weight there   mean weight there")
    texts = [TEXT, *(f"{word} {TEXT}" for word in FIRST_WORDS)]
    for other in (ids, *(model.encode(text) for text in texts)):
        count, mean = on_first(forward(other, model.weights, model.config))
        first = model.entries(other)[0]
        print(f"  {first:<13}  {f'{count} of {heads} heads':<38}{mean:.2f}")
    repeated = model.encode(REPEATED)
    entries = model.entries(repeated)
    rows = last_rows(forward(repeated, model.weights, model.config))
    later = entries.index(entries[0], 1)  # where the first token comes again
    print(
        f"  The same token twice, {REPEATED!r}: {entries[-1]} puts on average "
        f"{rows[..., 0].mean():.2f} of its weight on {entries[0]} at position 0, "
        f"and {rows[..., later].mean():.2f} on {entries[later]} at position {later}."
    )
    print("  It is the position: whatever comes first gets the weight.")

    # Step 3: what a head gets from the first token. The weights only say how
    # to mix the value vectors: each token adds weight x value to the head's
    # result. All lengths below are measured in one unit per head: the
    # average length of the other tokens' value vectors.
    lengths = np.stack([np.linalg.norm(layer.v, axis=-1) for layer in trace.layers])
    unit = lengths[..., 1:].mean(axis=-1)  # [layers, heads]
    weight_share = last_rows(trace)[..., 0]  # [layers, heads]
    shares = result_shares(trace)
    results = np.stack(  # [layers, heads]: the length of each head's result
        [np.linalg.norm(layer.head_outputs[:, -1], axis=-1) for layer in trace.layers]
    )
    parked, reading = weight_share > PARKED, weight_share < READING
    print("\nStep 3: what does a head get from the first token?")
    print(
        "  Its value vectors are short: in the median head, "
        f"{np.median(lengths[..., 0] / unit):.2f} times the average length of the "
        "other tokens' values."
    )
    print(
        "  Each token adds weight x value to a head's result. The length of the "
        f"first token's part is on average {shares.mean():.2f} of the lengths of "
        f"all tokens' parts added up, although it holds {weight_share.mean():.2f} "
        "of the weight."
    )
    print(
        "  So a head that parks its weight there returns little. In the "
        f"{parked.sum()} heads that put more than {PARKED} of the weight on the "
        "first token, the result is on average "
        f"{(results / unit)[parked].mean():.2f} times the average length of the "
        f"other tokens' values; in the {reading.sum()} heads that put less than "
        f"{READING} there, {(results / unit)[reading].mean():.2f} times."
    )
    print(
        "  A head that finds nothing to read parks its weight on the first token "
        "and adds little: an attention sink."
    )

    # Step 4: the first position stands out from the start. Its row of E_pos is
    # unlike the others, and in the residual stream its vector soon becomes far
    # longer than any other, through a few of its 768 numbers.
    position_lengths = np.linalg.norm(model.weights["wpe.weight"], axis=-1)
    print("\nStep 4: how does the first position stand out?")
    print(
        f"  In E_pos, row 0 is {position_lengths[0]:.1f} long; the other rows are "
        f"at most {position_lengths[1:].max():.1f} long, "
        f"{np.median(position_lengths[1:]):.1f} in the median."
    )
    print("  Length of each token's vector in the residual stream, after each block:")
    print(
        "    block  first token  other tokens  "
        "the first token's two numbers farthest from 0"
    )
    for i, layer in enumerate(trace.layers):
        length = np.linalg.norm(layer.h_out, axis=-1)
        first = layer.h_out[0]
        top = np.argsort(-np.abs(first))[:2]  # the two farthest from 0
        largest = ", ".join(f"number {n}: {first[n]:,.0f}" for n in top)
        print(
            f"    {i:>5}  {length[0]:>11,.0f}  {length[1:].min():>5,.0f} to "
            f"{length[1:].max():<5,.0f}  {largest}"
        )
    print(
        "  From block 1 on, two of the first token's 768 numbers make up almost all "
        "of its length. The last block shrinks it back to the size of the others."
    )

    print(
        "\nWhat this means:\n"
        "  - A large attention weight does not make a token important. Check what "
        "the head gets from there: weight x value.\n"
        "  - Most heads use the first position as a place to put weight. Xiao et "
        "al. (2023) found that models get much worse when a server keeps only the "
        "newest tokens of a long text in its cache, and that keeping the first "
        "few tokens as well largely repairs that: arXiv 2309.17453.\n"
        "  - Sun et al. (2024) found such huge numbers in many models, at the first "
        "token in GPT-2: they appear early, stay through the middle layers, shrink "
        "in the last ones and draw attention to their token: arXiv 2402.17762."
    )
    plot_heads(trace, model.entries(ids))
    plot_layers(weight_share, shares)


def plot_heads(trace: Trace, labels: list[str]) -> None:
    """The weights A of all 12 heads of LAYER; grey cells are masked."""
    fig = show.figure(13, 10)
    axes = fig.subplots(3, 4)
    image = None
    for head, ax in enumerate(axes.flat):
        image = show.heatmap(
            ax,
            trace.layers[LAYER].weights[head],
            labels,
            hide_future=True,
            numbers=False,
        )
        ax.set_title(f"head {head}", fontsize=10)
        if head % 4:
            ax.set_ylabel("")
        if head < 8:
            ax.set_xlabel("")
    if image is not None:
        show.colour_bar(fig, image, list(axes.flat), "attention weight")
    fig.suptitle(
        f"GPT-2 small, layer {LAYER}: the attention weights A of its 12 heads",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    show.save(fig, "attention_sink_heads")


def plot_layers(weight_share: Array, shares: Array) -> None:
    """Per layer: the first token's share of the weight and of the result."""
    fig = show.figure(9, 3.8)
    ax = fig.subplots()
    layers = np.arange(len(weight_share))
    width = 0.38
    ax.bar(
        layers - width / 2,
        weight_share.mean(axis=-1),
        width - 0.04,
        color=show.SERIES[0],
        label="share of the weight",
    )
    ax.bar(
        layers + width / 2,
        shares.mean(axis=-1),
        width - 0.04,
        color=show.SERIES[1],
        label="share of the result: its part's length over all parts' lengths",
    )
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean over the 12 heads")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title(
        "The first token gets more of the weight than of the result "
        "(last token, mean over the heads of each layer)"
    )
    show.save(fig, "attention_sink_layers")


if __name__ == "__main__":
    main()
