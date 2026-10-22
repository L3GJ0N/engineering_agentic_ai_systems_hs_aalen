# Working on these teaching examples

This repository contains the shareable Python examples for **Engineering Agentic AI Systems**, a master's course in computer science and data engineering at HS Aalen. It holds what students use: starters, exercises and guides.

## The private companion repository

Lecture planning, the slide decks and their build scripts, research code, instructor tools, exercise solutions and captured live responses live in a separate private repository. It is not part of this one and must never become visible through it:

- Do not import from it, link to it or name its paths here. Student code must run with this repository alone.
- Bring nothing over wholesale. When a prototype from the companion becomes teaching material, write a smaller, explained version here and re-verify every fact for publication.
- Material students need belongs here; material only the instructor needs belongs there. When unsure, ask.
- Ordinary example changes stay in this repository; work in the companion happens only when the user asks for it.

## Teaching priorities

- Explain what happens beneath the tools: HTTP, JSON, tokenization, inference, reasoning, evaluation and eventually agent loops. Keep the opening examples small enough to follow line by line.
- The course has no student compute budget and must work without a local GPU or paid subscription. Provide usable offline exercises where provider account access is unavailable. Do not promise that a provider's hosted access stays free.
- Students use Windows, Ubuntu/other Linux distributions and macOS. Give menu/button instructions first, with platform-specific keyboard shortcuts as optional aids. Treat WSL as its own Linux environment.
- Show hosted inference both ways: a direct Requests call that exposes the HTTP mechanics, and the same request through an SDK. Currently that is the OpenAI SDK for both services (`nvidia_inference.py` with `nvidia_inference_openai.py`, `openrouter_inference.py` with `openrouter_inference_openai.py`): NVIDIA's own example uses it, and OpenRouter's own `openrouter` SDK is deliberately not used. Keep each SDK example sending the same request as its Requests counterpart, with automatic retries switched off, so students can compare them line by line. Do not introduce frameworks or broad abstractions without a teaching reason or user request.

## Python and tooling

- Use uv for dependency and environment management. Do not introduce pip-based setup instructions.
- Keep the packaged `src/eaas_llm_inference/` layout and independent `llm_inference/pyproject.toml` project. Future examples may have their own uv projects.
- `token_lab/` (package `src/eaas_token_lab/`) is the second independent uv project: two short, explained scripts with everything fixed in the code, `tokenize_text.py` (a text alone) and `tokenize_chat.py` (after the chat template). They download only public tokenizer files, pinned to a commit, and never send a Hugging Face login (`token=False`). They run no model and contact no provider; the tests run offline. Label every token view as a local reconstruction unless a provider returned the tokens. Extend the series in the same style: one question per script, explained from the user's point of view.
- `exercises/` holds task sheets and their samples, Markdown and JSON only, with no uv project. An exercise's starter code is a public template repository of its own, such as [L3GJ0N/eaas_token_viewer_starter](https://github.com/L3GJ0N/eaas_token_viewer_starter); the task sheet links to it. Task sheets say what to deliver, never how it is graded.
- Use ordinary `uv sync` and `uv run` for student workflows. `--locked` is optional for strict verification/CI, not the required introductory command.
- Keep `pyproject.toml` and `uv.lock` consistent; use uv to update the lockfile.
- VS Code: each example project keeps its own `.vscode` folder (launch configurations with Windows and Linux/macOS interpreter paths, a `uv: sync` task, `settings.json` pointing to its `.venv`, extension recommendations). The multi-root workspace `engineering_agentic_ai_systems.code-workspace` at the repository root lists the repository and every project as folders; the root `.vscode` holds only extension recommendations, so no configuration appears twice. A new project gets its own `.vscode` folder and an entry in the workspace file.

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
