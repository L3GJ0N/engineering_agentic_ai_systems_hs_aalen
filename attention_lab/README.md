# Attention lab: the real matrices of a running model

Attention is usually explained with the matrices of one head for a short sentence, with made-up numbers that fit on a slide. This project computes the real ones. GPT-2 small, a complete language model published by OpenAI in 2019, runs on your laptop in a short NumPy program that keeps the matrices of every attention head. Each script answers one question with GPT-2's own numbers, in the terminal and as figures.

Everything runs on your laptop's CPU. You need no account, no key and no GPU, and no provider is contacted. The first run downloads three public files, 550 MB, from Hugging Face. Do that once at home, on a fast connection: in a full lecture hall, many downloads through one network are slow and can run into Hugging Face's limits for downloads without an account. Later runs reuse the files and work offline.

## First run

Open a terminal in the repository folder, then run:

```console
cd attention_lab
uv sync
uv run model-weights
```

`uv sync` installs NumPy, Matplotlib, three small Hugging Face libraries and the tools for the checks, about 200 MB. `model-weights` downloads the model and lists what its file holds.

## Three questions

Run them in this order, and read each file alongside: its comments explain every step.

```console
uv run model-weights
uv run one-head
uv run attention-sink
```

1. [model_weights.py](src/eaas_attention_lab/commands/model_weights.py): **where are W_Q, W_K, W_V and W_O?** The model file holds 160 named arrays. W_Q, W_K and W_V of a block sit side by side in one 768 × 2304 matrix, and the four matrices of a block hold exactly 4d² = 2,359,296 weights. The file also holds what diagrams rarely show: biases, LayerNorm, the feed-forward networks with 46 % of all weights, and a stored causal mask. One thing is missing: the output layer has no matrix of its own.
2. [one_head.py](src/eaas_attention_lab/commands/one_head.py): **one head by hand.** Read the docstring of [gpt2.py](src/eaas_attention_lab/gpt2.py) first: it explains LayerNorm and the residual stream, which attention diagrams usually leave out. The queries, keys and values of head 4 in layer 5 for "the cat sat on it", the scores S, the mask, the weights A and the head's result, each step checked against the whole model. S is not symmetric; dividing by √64 keeps the order within each row but changes the weights; masking after the softmax leaves every row but the last summing to less than 1.
3. [attention_sink.py](src/eaas_attention_lab/commands/attention_sink.py): **why most heads look at the first token.** In 114 of the 144 heads, " it" puts its largest weight on the first token, and in about as many when a word of the sentence comes first. The first token's value vectors are short, so a head gets little from there: it is a place to put weight when there is nothing to read. The script ends with why this matters for reading attention maps, and for servers with long conversations.

To try another text, layer or head, change `TEXT`, `LAYER` or `HEAD` at the top of a script, save, and run it again.

## What you see

- Each script prints its steps in the terminal and saves its figures as PNG files in `outputs/`, which Git ignores; open them in VS Code. A run overwrites its figures: to compare two settings, copy a figure first.
- In every matrix of scores or weights, row i belongs to the token that reads, its query, and column j to the token that is read, its key. Weights are blue on a fixed scale from 0 to 1, so a colour means the same weight in every figure. Scores are red below 0, light grey near 0 and blue above. Darker grey cells are hidden by the mask. In the figure of `one_head.py`, the cells also show their values for texts of up to 8 tokens, and its terminal tables of S and A hold them for any length; the 12 small panels of `attention_sink.py` show colours only.
- Layers and heads count from 0, as Python does: layer 5 is the sixth block. Vocabulary entries are shown as in [token_lab](../token_lab/README.md): `Ġ` stands for a space.
- GPT-2 adds no start token to a text. The scripts put its only special token, `<|endoftext|>`, in front; `attention_sink.py` shows why the first token matters.

## How the code is organized

```text
src/eaas_attention_lab/
    gpt2.py                 library: GPT-2's forward pass
    model_files.py          library: download and read the model's files
    show.py                 library: tables and figures
    commands/               what you run, one question per file
        model_weights.py    uv run model-weights
        one_head.py         uv run one-head
        attention_sink.py   uv run attention-sink
```

A command holds its settings at the top and one function, `main()`, that answers its question in numbered steps. The measurements that answer the question are written out there, so that you can read and change them. Everything a command reuses, the model, its files and the figures, comes from the library.

## The model in NumPy

[gpt2.py](src/eaas_attention_lab/gpt2.py) is the whole model, about 100 lines: the embedding lookup, 12 blocks of attention and a feed-forward network, and the output layer. It includes the two parts that attention diagrams usually leave out, LayerNorm and the residual connections, and `forward()` returns a `Trace` with the queries, keys, values, scores and weights of every head in every layer, and the residual stream after every step. Hugging Face Transformers returns only part of this: the keys and values as its cache, and on request the weights and the residual stream between blocks, but not the queries, the scores before the softmax, each head's result or the residual stream inside a block. The test `tests/test_reference.py` shows that it computes the same numbers as the reference implementation; see [Checks](#checks).

[model_files.py](src/eaas_attention_lab/model_files.py) downloads and reads the three files, pinned to one revision of the repository [openai-community/gpt2](https://huggingface.co/openai-community/gpt2). Its model card gives the licence as MIT; OpenAI's own [GPT-2 repository](https://github.com/openai/gpt-2) uses a modified MIT licence, which also asks you to use GPT-2 responsibly and to say when content was created with it. [show.py](src/eaas_attention_lab/show.py) prints matrices as tables and draws them as figures.

The downloaded files are plain files in `downloads/gpt2/607a30d783df/`, which Git ignores: `config.json` holds the hyperparameters, `tokenizer.json` the vocabulary and `model.safetensors` the weights. If a download breaks off, run the command again: finished files are kept, but the weights file starts over from the beginning. If the program was killed rather than stopped, a leftover file ending in `.incomplete` may remain in `downloads/gpt2/607a30d783df/.cache/huggingface/download/`; it is never used, and you can delete it. Copying the folder `downloads` from a classmate's `attention_lab` works as well.

Hugging Face may print a warning about unauthenticated requests. It is harmless: these files are public, and the scripts never send a Hugging Face login, even if one is stored on your computer.

## GPT-2 and current models

GPT-2 small has the plain design that attention diagrams draw: 12 identical blocks of attention and a feed-forward network, every head with its own queries, keys and values, and position vectors added once, to the input. Current open models keep the core, queries meeting keys and a softmax over their scores weighting the values, but change much around it:

- Most do not add position vectors to the input. They rotate each query and key by an angle that depends on its position (rotary position embeddings, RoPE), so that the score of two tokens depends on their distance, not on where they stand.
- Many let several query heads share one key head and one value head (grouped-query attention), which makes the cache of keys and values smaller.
- Some keep only a few attention layers. The course model, Nemotron 3.5 Lightning, has 52 layers. Only 6 are attention layers, in which 32 query heads share 2 key and value heads and no position encoding is applied. 23 are Mamba-2 layers, which read the tokens one after another and carry a state of fixed size instead of a growing cache of keys and values. 23 are mixture-of-experts layers, in which each token passes through 6 of 128 expert networks and one shared network. Hence the 30B-A3B in its full name: about 30 billion weights, of which about 3 billion work on each token, while GPT-2 small has 124 million.

## Debug in VS Code

Open the course workspace, `engineering_agentic_ai_systems.code-workspace` in the repository folder, with **File > Open Workspace from File**, or this folder alone with **File > Open Folder**. Both use this project's configurations **Attention: Model weights (local)**, **Attention: One head (local)** and **Attention: Sink (local)** and its own Python environment. Select the **Run and Debug** view, choose one and use **Start Debugging**; see the [debugging guide](../llm_inference/debugging.md) for the procedure. If uv warns that `VIRTUAL_ENV` does not match the project environment, the window shows the repository folder instead of the workspace: see [Several projects, one VS Code workspace](../README.md#several-projects-one-vs-code-workspace).

A good breakpoint: in `gpt2.py`, the line `attention_weights = softmax(scores + causal_mask(len(h)))` in `block()`, to see the scores of all 12 heads of a layer. To stop in layer 5 only, right-click left of the line number, choose **Add Conditional Breakpoint** and enter `i == 5`. Then start **Attention: One head (local)**; it stops at the script's first line, so use **Continue** once. In the **Debug Console**, `scores[4].round(2)` prints the scores of head 4.

## Checks

```console
uv run pytest
uv run mypy
uv run ruff check
```

The tests need no network. `test_gpt2.py` runs the forward pass on a tiny model with random weights. `test_scripts.py` runs the three scripts on the files downloaded by an earlier run and compares the printed numbers with the known ones; before the first download it is skipped.

`test_reference.py` compares `gpt2.py` with Hugging Face Transformers, the library most Python code uses to run such models. It needs PyTorch and Transformers, which no script needs, and is skipped without them. To run it:

```console
uv run --group reference pytest tests/test_reference.py
```

The first time, this installs PyTorch, from its package index for CPUs, and Transformers: 400 to 850 MB, depending on the system. The next `uv sync` removes them from the environment; uv keeps a copy in its cache, which `uv cache clean torch transformers` deletes. On macOS, PyTorch publishes current versions only for Apple silicon with macOS 14 or later; on other Macs, uv reports that no wheel fits, and the check cannot run. The scripts run on these Macs too.
