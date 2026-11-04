# Engineering Agentic AI Systems, HS Aalen

Python teaching examples for the master's course **Engineering Agentic AI Systems**. The examples build understanding of hosted LLM inference and the systems around a model, beginning with small, readable Python programs.

## Current examples

[llm_inference](llm_inference/README.md) is an independent, packaged uv project. It includes:

- One direct NVIDIA Build inference request using Requests.
- An offline HTTP request inspection that needs no account or key.
- A capture of one complete live exchange, printed and saved without credentials.
- The same request sent through the OpenAI SDK, to compare with the direct Requests call.
- The same model through OpenRouter, a service that forwards requests to model providers, again as a direct Requests call and through the OpenAI SDK.
- Detailed HTTP, JSON, credential configuration, NVIDIA account and OpenRouter account walkthroughs.
- VS Code launch and attach configurations for Windows, Linux and macOS, with a debugging guide.

[token_lab](token_lab/README.md) is a second uv project. It shows the tokens a model reads for a prompt, using the tokenizer and chat template the model's publisher released, entirely on your laptop. It includes:

- Two short, explained scripts: the tokens of the prompt alone, and the tokens the model actually reads after the chat template, with the special tokens marked.
- How to find a model's tokenizer on Hugging Face.
- A guide to what a hosted API lets you see of a model and how to find out, with exercises: [what_you_can_access.md](token_lab/what_you_can_access.md).

[attention_lab](attention_lab/README.md) is a third uv project. It runs a real language model, GPT-2 small, on your laptop's CPU and shows its real weights and attention matrices, which diagrams only sketch with made-up numbers. It includes:

- Three short, explained scripts: where W_Q, W_K, W_V and W_O are in the model file; one attention head computed by hand for "the cat sat on it"; and why most heads put their weight on the first token.
- The whole model in about 100 lines of NumPy, keeping the matrices of every attention head, and checked against the reference implementation.
- Figures of the matrices, saved as PNG files.

## Exercises

- [Token viewer](exercises/token_viewer/README.md): a group exercise in class. Each group builds a token viewer for chat requests, in Python, starting from its own copy of a template repository. The task sheet says what to deliver and how the presentation works.

## Start here

Get the repository: on its [GitHub page](https://github.com/L3GJ0N/engineering_agentic_ai_systems_hs_aalen), select **Code → Download ZIP** and extract the archive, or clone it with Git:

```console
git clone https://github.com/L3GJ0N/engineering_agentic_ai_systems_hs_aalen.git
cd engineering_agentic_ai_systems_hs_aalen
```

The repository folder is the one that directly contains `README.md` and `llm_inference`. A downloaded archive extracts to `engineering_agentic_ai_systems_hs_aalen-main`; on Windows, **Extract All** can place it inside a second folder of the same name.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open a terminal in the repository folder, then run:

```console
cd llm_inference
uv sync
uv run inspect-http-request
```

The offline example prepares and displays a fake request without sending it. For hosted inference, follow the [NVIDIA walkthrough](llm_inference/nvidia_walkthrough.md) and configure the credential in your private `llm_inference/.env`. The tracked `.env.example` declares empty fields only. Then run from `llm_inference`:

```console
uv run nvidia-inference
```

To send the same prompt to the same model through OpenRouter, follow the [OpenRouter walkthrough](llm_inference/openrouter_walkthrough.md), add that key to the same `.env` and run `uv run openrouter-inference`.

No local GPU is required for these hosted inference examples. Provider access and availability depend on the student's account; the offline example works independently of that access.

To see the tokens the model reads, run from the repository folder; the first run downloads about 17 MB of public tokenizer files, and no account is needed:

```console
cd token_lab
uv sync
uv run tokenize-text
uv run tokenize-chat
```

`tokenize-text` shows the tokens of the prompt alone, `tokenize-chat` the tokens the model actually reads after the chat template; each is one short file.

To see a model's real weights and attention matrices, run from the repository folder. The first run downloads 550 MB of public model files, so run it once on a fast connection, for example at home; no account is needed:

```console
cd attention_lab
uv sync
uv run model-weights
uv run one-head
uv run attention-sink
```

Each command answers one question with GPT-2 small's own numbers, in the terminal and as figures in `attention_lab/outputs/`.

## Several projects, one VS Code workspace

The repository holds three independent uv projects: `llm_inference`, `token_lab` and `attention_lab`. Each has its own `pyproject.toml` and lockfile, its own Python environment in its own `.venv` folder, and its own VS Code settings and debug configurations in its own `.vscode` folder. They share nothing: `uv sync` in `token_lab` installs only what `token_lab` needs. There is no Python project at the repository's root, so run uv commands inside a project's folder, as in [Start here](#start-here). `uv run` always uses the environment of the project it runs in; you never need to activate one.

The file `engineering_agentic_ai_systems.code-workspace` in the repository folder ties the projects together for VS Code. Open it, not the folder:

1. Select **File > Open Workspace from File** and choose `engineering_agentic_ai_systems.code-workspace`. If you opened the repository folder with **File > Open Folder**, VS Code may offer the workspace in a notification; choose **Open Workspace**.
2. Check the **Explorer**: it shows four folders, **repository** (this README and AGENTS.md), **llm_inference**, **token_lab** and **attention_lab**. A single folder tree means that the window shows the folder, not the workspace.

Only the workspace gives each project its own Python environment in the editor and lists the debug configurations of every project. In a window opened on the repository folder, the editor checks all code against one environment and may mark imports as missing, **Run and Debug** offers no configurations, and **Terminal > New Terminal** starts in the repository folder without asking for a project. Opening one project folder alone, such as `attention_lab`, works as well: its settings and configurations come with it.

If uv prints `warning: VIRTUAL_ENV=... does not match the project environment path ... and will be ignored`, the terminal has another project's environment activated. The warning is harmless: uv uses the right environment anyway. Run `deactivate`, or open a new terminal in the project's folder. Do not add `--active`, as the warning suggests: that would install this project's packages into the other project's environment. The repository's VS Code settings stop VS Code from activating environments in new terminals, so the warning appears only after an environment was activated by hand or by your own VS Code settings.

## Debug in VS Code

In the course workspace, each example project uses its own Python environment, `uv sync` task and debug configurations, from its own `.vscode` folder:

- **llm_inference:** **LLM: Inspect HTTP (offline)**, **LLM: NVIDIA inference (live)**, **LLM: NVIDIA inference via OpenAI SDK (live)**, **LLM: OpenRouter inference (live)**, **LLM: OpenRouter inference via OpenAI SDK (live)** and **LLM: Attach to uv/debugpy (5678)**.
- **token_lab:** **Tokens: Tokenize text (local)** and **Tokens: Tokenize chat (local)**.
- **attention_lab:** **Attention: Model weights (local)**, **Attention: One head (local)** and **Attention: Sink (local)**.

Select the **Run and Debug** view, choose a configuration and use **Start Debugging**; the list names the project after each configuration. The launch configurations run `uv sync` in the example project before starting its Python interpreter. See the [debugging guide](llm_inference/debugging.md) for the full procedure, platform differences and breakpoint exercises.

## Repository structure

```text
engineering_agentic_ai_systems_hs_aalen/
    README.md
    AGENTS.md
    engineering_agentic_ai_systems.code-workspace   VS Code workspace: all projects
    .gitignore
    .gitattributes
    .vscode/                     extension recommendations; no environments in terminals
    llm_inference/
        pyproject.toml
        uv.lock
        .python-version
        .env.example             empty configuration fields; no credential
        .vscode/                 launch configurations for this project alone
        src/eaas_llm_inference/
        README.md
        http_and_requests.md
        nvidia_walkthrough.md
        openrouter_walkthrough.md
        debugging.md
    token_lab/
        pyproject.toml
        uv.lock
        .python-version
        .vscode/                 launch configurations for this project alone
        src/eaas_token_lab/
        tests/
        README.md
        what_you_can_access.md
    attention_lab/
        pyproject.toml
        uv.lock
        .python-version
        .vscode/                 launch configurations for this project alone
        src/eaas_attention_lab/
        tests/
        README.md
    exercises/
        token_viewer/
            README.md            the task sheet
            target_page.png      the picture on the task sheet
            samples/             the course request's messages as JSON
```

Each example owns its Python environment and dependencies. There is no root Python project; run uv commands from the example's folder, `llm_inference`, `token_lab` or `attention_lab`. An exercise's folder holds its task sheet and samples; the code a group starts from is a template repository of its own. Additional examples can be added as the course develops.

This repository contains the shareable examples. Lecture slides and other course-planning materials are maintained separately.
