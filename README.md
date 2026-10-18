# Engineering Agentic AI Systems — HS Aalen

Python teaching examples for the master's course **Engineering Agentic AI Systems**. The examples build understanding of hosted LLM inference and the systems around a model, beginning with small, readable Python programs.

## Current example

[llm_inference](llm_inference/README.md) is an independent, packaged uv project. It includes:

- One direct NVIDIA Build inference request using Requests.
- An offline HTTP request inspection that needs no account or key.
- A capture of one complete live exchange, printed and saved without credentials.
- The same request sent through the OpenAI SDK, to compare with the direct Requests call.
- The same model through OpenRouter, a service that forwards requests to model providers, again as a direct Requests call and through the OpenAI SDK.
- Detailed HTTP, JSON, credential configuration, NVIDIA account and OpenRouter account walkthroughs.
- VS Code launch and attach configurations for Windows, Linux and macOS, with a debugging guide.

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

## Debug in VS Code

Open this repository folder or `llm_inference` itself in VS Code. Both have configurations for **LLM: Inspect HTTP (offline)**, **LLM: NVIDIA inference (live)**, **LLM: NVIDIA inference via OpenAI SDK (live)**, **LLM: OpenRouter inference (live)**, **LLM: OpenRouter inference via OpenAI SDK (live)** and **LLM: Attach to uv/debugpy (5678)**.

Select the **Run and Debug** view, choose a configuration and use **Start Debugging**. The launch configurations run `uv sync` in the example project before starting its Python interpreter. See the [debugging guide](llm_inference/debugging.md) for the full procedure, platform differences and breakpoint exercises.

## Repository structure

```text
engineering_agentic_ai_systems_hs_aalen/
    README.md
    AGENTS.md
    .gitignore
    .gitattributes
    .vscode/                     launch configurations for the repository root
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
```

Each example owns its Python environment and dependencies. There is no root Python project; run uv commands from `llm_inference` for the current example. Additional examples can be added as the course develops.

This repository contains the shareable examples. Lecture slides and other course-planning materials are maintained separately.
