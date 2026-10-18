# Working on these teaching examples

This repository contains the shareable Python examples for **Engineering Agentic AI Systems**, a master's course in computer science and data engineering at HS Aalen. Keep work in this repository; the separate lecture-planning workspace is outside the scope of ordinary example changes.

## Teaching priorities

- Explain what happens beneath the tools: HTTP, JSON, tokenization, inference, reasoning, evaluation and eventually agent loops. Keep the opening examples small enough to follow line by line.
- The course has no student compute budget and must work without a local GPU or paid subscription. Provide usable offline exercises where provider account access is unavailable. Do not promise that a provider's hosted access stays free.
- Students use Windows, Ubuntu/other Linux distributions and macOS. Give menu/button instructions first, with platform-specific keyboard shortcuts as optional aids. Treat WSL as its own Linux environment.
- Show hosted inference both ways: a direct Requests call that exposes the HTTP mechanics, and the same request through an SDK. Currently that is the OpenAI SDK for both services (`nvidia_inference.py` with `nvidia_inference_openai.py`, `openrouter_inference.py` with `openrouter_inference_openai.py`): NVIDIA's own example uses it, and OpenRouter's own `openrouter` SDK is deliberately not used. Keep each SDK example sending the same request as its Requests counterpart, with automatic retries switched off, so students can compare them line by line. Do not introduce frameworks or broad abstractions without a teaching reason or user request.

## Python and tooling

- Use uv for dependency and environment management. Do not introduce pip-based setup instructions.
- Keep the packaged `src/eaas_llm_inference/` layout and independent `llm_inference/pyproject.toml` project. Future examples may have their own uv projects.
- Use ordinary `uv sync` and `uv run` for student workflows. `--locked` is optional for strict verification/CI, not the required introductory command.
- Keep `pyproject.toml` and `uv.lock` consistent; use uv to update the lockfile.
- Preserve VS Code configurations at both the repository root and example-project root. Root launch/task paths must point to `llm_inference`, including the correct Windows versus Linux/macOS interpreter paths.

## Credentials and execution

- Never put a real API key or a "paste your key here" placeholder into Python source, launch configurations or any other committed code.
- The live examples read a private, ignored `llm_inference/.env` explicitly with `dotenv_values(..., interpolate=False)`, each selecting its own field (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`). Keep `.env.example` limited to empty field declarations. Do not switch to `getpass()` or process-environment injection unless the user requests that change.
- Do not inspect, print, commit or include private `.env` contents in artifacts. Git ignore rules are not protection for arbitrary folder/ZIP sharing.
- The offline HTTP inspector intentionally uses a fake classroom credential. It must never retrieve a real credential or send a request.
- Verify with the offline example and mocked responses when possible. Do not run real inference merely to check a refactor; a live request must be part of the user's requested work.

## Scope and validation

- Pre-existing locally ignored material is outside the examples repository's tracked content. Leave it untouched unless the user explicitly brings it into scope.
- Keep imports, internal Markdown links and launch configurations valid after moving files.
- Verify changes with checks appropriate to their impact; avoid tests that merely restate trivial implementation.
- Distinguish a successful HTTP response from a correct model answer, and local Python debugging from remote model execution.
- Future topics include Hugging Face, token inspection, reasoning output, systematic evaluation and agent loops. These are directions, not a request to implement them all at once.
- This repository is public on GitHub as [L3GJ0N/engineering_agentic_ai_systems_hs_aalen](https://github.com/L3GJ0N/engineering_agentic_ai_systems_hs_aalen), with `origin` pointing there. Anything committed becomes public once pushed: push only when the user asks, and never commit credentials, `.env` files, captured responses or personal e-mail addresses. Commit with a GitHub noreply address.
