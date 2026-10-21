# Engineering Agentic AI Systems — HS Aalen

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

## Debug in VS Code

Open the course workspace: **File > Open Workspace from File**, then select `engineering_agentic_ai_systems.code-workspace` in the repository folder. If you open the repository folder with **File > Open Folder** instead, VS Code offers to open the workspace in a notification; choose **Open Workspace**. The **Explorer** then shows three folders: **repository** (this README and AGENTS.md), **llm_inference** and **token_lab**. Each example project uses its own Python environment, `uv sync` task and debug configurations, from its own `.vscode` folder:

- **llm_inference:** **LLM: Inspect HTTP (offline)**, **LLM: NVIDIA inference (live)**, **LLM: NVIDIA inference via OpenAI SDK (live)**, **LLM: OpenRouter inference (live)**, **LLM: OpenRouter inference via OpenAI SDK (live)** and **LLM: Attach to uv/debugpy (5678)**.
- **token_lab:** **Tokens: Tokenize text (local)** and **Tokens: Tokenize chat (local)**.

Select the **Run and Debug** view, choose a configuration and use **Start Debugging**; the list names the project after each configuration. The launch configurations run `uv sync` in the example project before starting its Python interpreter. You can also open a single project folder, such as `llm_inference`, with **File > Open Folder**; its configurations are the same. See the [debugging guide](llm_inference/debugging.md) for the full procedure, platform differences and breakpoint exercises.

## Repository structure

```text
engineering_agentic_ai_systems_hs_aalen/
    README.md
    AGENTS.md
    engineering_agentic_ai_systems.code-workspace   VS Code workspace: all projects
    .gitignore
    .gitattributes
    .vscode/                     extension recommendations
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
```

Each example owns its Python environment and dependencies. There is no root Python project; run uv commands from the example's folder, `llm_inference` or `token_lab`. Additional examples can be added as the course develops.

This repository contains the shareable examples. Lecture slides and other course-planning materials are maintained separately.
