# Token lab: how a model's tokenizer sees your prompt

A model never reads your text. It reads token IDs, produced by its own tokenizer after its chat template has written your messages into one text. This project shows that step for the course model, Nemotron 3.5 Lightning, with the tokenizer and chat template its publisher released.

Everything runs on your laptop. You need no account, no key and no GPU; no model runs and no provider is contacted. The first run downloads about 17 MB of public files from Hugging Face; later runs reuse them.

## Two steps

Open a terminal in the repository folder, then run:

```console
cd token_lab
uv sync
uv run tokenize-text
uv run tokenize-chat
```

Two short files with everything fixed in the code. Read them in this order; their comments explain each choice and name the alternatives.

1. [tokenize_text.py](src/eaas_token_lab/tokenize_text.py): **the text alone.** The NVIDIA example sends "Explain tokenization in one sentence." in its JSON body; this file shows the 8 tokens that text becomes with the model's tokenizer, and decodes them back. It needs only `tokenizer.json`, and explains how to find a model's repository on Hugging Face and why its revision is pinned.
2. [tokenize_chat.py](src/eaas_token_lab/tokenize_chat.py): **what the model actually reads.** The server writes the chat messages into one text with the model's chat template before encoding: 24 tokens. The file explains why chat models need these roles, marks the special tokens, and why the downloaded template runs in a sandbox.

To try another text, change `TEXT` in `tokenize_text.py` or the content in `MESSAGES` in `tokenize_chat.py`, save, and run the script again. [what_you_can_access.md](what_you_can_access.md) has exercises.

Hugging Face may print a warning about unauthenticated requests. It is harmless: these files are public, and the scripts never send a Hugging Face login, even if one is stored on your computer.

## What you see

- One row per token: its **ID**, which is all the model receives, and the **vocabulary entry** it stands for. In entries, `Ġ` stands for a space and `Ċ` for a newline.
- Step 1: the text alone is 8 tokens. Step 2: after the chat template the model reads 24. The template adds an empty system turn, the role markers and the start of the assistant's turn.
- Step 2 marks the **special tokens**. Only the role markers `<|im_start|>` and `<|im_end|>` are special; the role names `system`, `user` and `assistant` are ordinary vocabulary entries.
- Both steps decode the IDs back and confirm that the original text comes out.

This is a **local reconstruction**. No provider returns the token IDs it used; why the reconstruction is still trustworthy, and what you can and cannot see behind a hosted API, is explained in [what_you_can_access.md](what_you_can_access.md).

The downloaded files are plain files: open `downloads/nemotron-3.5-lightning/bee7596271d1/tokenizer.json` to see the whole vocabulary, and `chat_template.jinja` beside it for the template. `downloads/` is ignored by Git.

## Find a model's tokenizer on Hugging Face

The two scripts are fixed to one repository. To check that choice, or to look at another open-weight model, find its repository in five steps. The labels were checked on 19 October 2026.

1. **Start from the provider's model page.** The model ID a provider uses, such as `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA Build, is not a Hugging Face name. The model page usually links to the published model: on [NVIDIA Build's page for the model](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b), the link **Hugging Face** under **Release Date** leads to the repository. OpenRouter names a repository in the field `hugging_face_id` of its public [model list](https://openrouter.ai/api/v1/models), a large JSON file; for our model it names the BF16 variant.
2. **Or search.** On [huggingface.co](https://huggingface.co/), type the model's name, for example `nemotron-3.5-lightning`, into the search box at the top. Many results are copies, conversions or fine-tuned versions by other users; on 19 October 2026 the first result was one of them. Take a repository owned by the publisher: the part of the name before the slash, here `nvidia`.
3. **Choose the variant.** One model often has several repositories. NVIDIA publishes `…-NVFP4`, `…-BF16` and `…-Base-BF16`, plus `…-NVFP4-DSpark` and `…-NVFP4-DFlash`. Choose the variant the provider runs: NVIDIA Build links the NVFP4 repository, and OpenRouter labels the endpoint NVFP4. Its tokenizer files are identical to those of the BF16 repository. The Base repository holds the model before its chat training and has no chat template; the DSpark and DFlash repositories are helper models for faster generation and contain no tokenizer.
4. **Check the files.** Open the tab **Files and versions** and look for `tokenizer.json` and `chat_template.jinja`. Some models keep their chat template inside `tokenizer_config.json`, under the key `chat_template`, instead of in a separate file. On the tab **Model card**, the licence tag, here **License: openmdw-1.1**, shows the terms; a form asking you to agree before access means the files need a Hugging Face login, which these scripts do not send.
5. **Note the name and the revision.** The repository name is the text after `huggingface.co/` in the address bar. For the revision, open **Files and versions**, then **History: … commits**, and click the newest commit: the address now ends with its full 40-character hash. Or open `https://huggingface.co/api/models/<repository>` in the browser and copy the value of `"sha"`.

## Debug in VS Code

Open the course workspace, `engineering_agentic_ai_systems.code-workspace` in the repository folder, with **File > Open Workspace from File**, or this folder alone with **File > Open Folder**. Both use this project's configurations **Tokens: Tokenize text (local)** and **Tokens: Tokenize chat (local)** and its own Python environment. Select the **Run and Debug** view, choose one and use **Start Debugging**; see the [debugging guide](../llm_inference/debugging.md) for the procedure. Good breakpoints in `tokenize_chat.py`: the line `encoding: Encoding = tokenizer.encode(text, add_special_tokens=False)`, to read `text` before it becomes tokens, and the `for` loop over the tokens. These scripts handle no keys, so the **Variables** panel shows nothing secret.

## Checks

```console
uv run pytest
uv run mypy
uv run ruff check
```

The tests need no network: they run both scripts on the files downloaded by an earlier run and compare the printed IDs with the known ones. Before the first download they are skipped.
