# LLM inference examples

This standalone **packaged uv project** starts with one NVIDIA inference request and an offline HTTP inspection. It follows a `uv init --package` scaffold, with Python code in `src/eaas_llm_inference/`. We will add modules for other providers, tokenization, reasoning and evaluation here as the course develops. All examples share one declared set of dependencies and one local Python environment.

Start with [HTTP, configuration and Requests](http_and_requests.md), then follow [NVIDIA account setup and inference](nvidia_walkthrough.md). [Debugging in VS Code](debugging.md) shows how to pause the examples and inspect them line by line. These guides and the scripts are self-contained; provider comparisons and tokenizer viewers will be added as later examples.

## Run the project

Check that uv is available:

```console
uv --version
```

If it is missing, use [Astral's installation instructions](https://docs.astral.sh/uv/getting-started/installation/). On Windows, their documented WinGet option is `winget install --id=astral-sh.uv -e`. Open a new terminal afterward if the command is not yet found.

From the `engineering_agentic_ai_systems_hs_aalen` repository root:

```console
cd llm_inference
uv sync
uv run inspect-http-request
```

`uv sync` prepares the local `.venv` with the resolved dependencies and installs this project in **editable mode**: imports use the Python files in `src/`, so edits to their code take effect on the next run. It also installs the three command entry points declared in `pyproject.toml`. `uv run` prepares the environment as needed and runs a command using it. No manual environment activation is needed. These commands update `uv.lock` when project requirements require it; a new package release alone does not cause an upgrade. Add `--locked` when an outdated or missing lockfile should be an error, for example in CI. Initial setup needs network access when Python or packages are not already available. The inspection script itself sends no network request and needs no key. [uv environment and lock behavior](https://docs.astral.sh/uv/concepts/projects/sync/).

The Python code retrieves the credential from your private **`.env`** in the project root, beside `pyproject.toml`. The tracked [.env.example](.env.example) declares only the configuration field, with an empty value:

```dotenv
NVIDIA_API_KEY=
```

Configure the value only in your private, untracked `.env`. Python source uses the configuration field's name and the value read at runtime; it contains no key literal or paste-here placeholder. If you already configured `.env`, keep using it. In Windows, check that the filename is `.env`, not `.env.txt`. Then run:

```console
uv run nvidia-inference
```

The program reads the file, sends one request to NVIDIA, and prints the HTTP status and generated answer for a normal completed response. The model runs remotely; no GPU or model download is needed on your laptop. The current prompt asks for a one-sentence explanation of tokenization. Change that string in the script for your next experiment.

To see everything that came back, not only the answer, run:

```console
uv run capture-nvidia-response
```

It sends the same single request, then prints the request as it was sent and the complete response: status, headers, timing and the full JSON body. It also saves this capture under a timestamped name in `outputs/`, which Git ignores. Header values that can carry credentials, such as `Authorization`, are replaced by `<removed>`, and the script refuses to print or save anything that still contains your key, so a capture can be shown in class. Each run is one more live request. [What comes back?](http_and_requests.md#7-what-comes-back) explains every part.

Keep `.env` in this project root. The code explicitly resolves that root from its location in `src/eaas_llm_inference/`; see [the configuration explanation](http_and_requests.md#52-read-our-local-configuration-one-line-at-a-time). Neither command needs a key in a command-line argument, and the live starter does not use uv's `--env-file` option: its Python code explicitly reads `.env` with `dotenv_values`.

## Understand the project files

```text
llm_inference/
    pyproject.toml           project metadata, dependencies and Ruff settings
    uv.lock                  resolved package versions and distributions
    .python-version          preferred Python version: 3.13
    .gitignore               files excluded from ordinary Git tracking
    .env.example             configuration field with an empty value
    http_and_requests.md     the underlying mechanisms
    nvidia_walkthrough.md    account setup and NVIDIA details
    debugging.md             debugging the examples in VS Code
    README.md                this guide
    .venv/                   local Python environment created by uv
    .env                     private key, created by you
    outputs/                 key-free response captures, created when needed
    .vscode/                 debugger, uv task and extension recommendations
    src/
        eaas_llm_inference/
            __init__.py                 marks the Python package
            nvidia_inference.py         one live inference request
            inspect_http_request.py     inspect a fake request without sending it
            capture_nvidia_response.py  print and save a key-free live exchange
```

The project accepts Python 3.11 or newer; `.python-version` selects 3.13 for the usual course workflow. uv can use a suitable existing interpreter or obtain one when needed. [Python selection](https://docs.astral.sh/uv/concepts/python-versions/#python-version-files).

`pyproject.toml` describes the allowed dependencies. `uv.lock` records their resolved versions, including dependencies of dependencies. Share both files so another student can reconstruct the environment. `.venv` is the installation on your machine; do not distribute it. [Project structure](https://docs.astral.sh/uv/concepts/projects/layout/).

**`.venv` and `.env` have different jobs.** `.venv` holds Python and installed packages; `.env` holds this example's configuration. The configuration file is plain text. Keep it out of submissions, shared folders and ZIPs; `.gitignore` controls Git tracking, not encryption or arbitrary file sharing.

`[build-system]` in `pyproject.toml` selects `uv_build`, the build backend from the generated scaffold. It makes our source package installable. This is local Python packaging; it does not publish anything or deploy a model. The distribution name is `eaas-llm-inference`, while Python imports use the module name `eaas_llm_inference`.

`[project.scripts]` connects terminal commands to ordinary Python functions:

| Terminal command | Function called |
|---|---|
| `inspect-http-request` | `main()` in [inspect_http_request.py](src/eaas_llm_inference/inspect_http_request.py) |
| `nvidia-inference` | `main()` in [nvidia_inference.py](src/eaas_llm_inference/nvidia_inference.py) |
| `capture-nvidia-response` | `main()` in [capture_nvidia_response.py](src/eaas_llm_inference/capture_nvidia_response.py) |

For example, `nvidia-inference = "eaas_llm_inference.nvidia_inference:main"` means "import this module and call its `main` function." The launcher is created during installation. `uv run` selects the environment and runs that launcher. NVIDIA inference starts only when the function reaches `requests.post(...)` and NVIDIA accepts the request.

You can also run the modules explicitly. Python's `-m` option finds an installed module and executes it; the `if __name__ == "__main__"` block then calls `main()`:

```console
uv run python -m eaas_llm_inference.inspect_http_request
uv run python -m eaas_llm_inference.nvidia_inference
uv run python -m eaas_llm_inference.capture_nvidia_response
```

The key-loading path is intentionally tied to this course checkout and uv's editable installation. Keep `.env` in the project root. A separately distributed wheel would need its own configuration-location policy.

## Debug in VS Code

The repository folder and this project folder both provide VS Code launch configurations: **LLM: Inspect HTTP (offline)**, **LLM: NVIDIA inference (live)** and **LLM: Attach to uv/debugpy (5678)**. The short route:

1. Install Microsoft's **Python** and **Python Debugger** extensions (the workspace recommends them) and run `uv sync` once.
2. Open [inspect_http_request.py](src/eaas_llm_inference/inspect_http_request.py) and click the editor gutter beside a line to set a breakpoint.
3. Select the **Run and Debug** view, choose **LLM: Inspect HTTP (offline)** and click **Start Debugging**. The debugger first pauses on line 1; click **Continue** to reach your breakpoint.

The debugger controls only the Python process on your laptop; the model still runs at NVIDIA. **In a paused live session, Variables > Locals shows your API key in plain text**, so keep that view off shared screens and screenshots.

The [debugging guide](debugging.md) explains what a debugger does underneath, every step and button with shortcuts for Windows, Linux and macOS, WSL, stepping into Requests, attaching to a process started by uv, and inspecting a live response safely.

## How the scaffold was chosen

The project started from a scaffold generated with **uv 0.8.19**, then added the examples, dependency requirements and guides. For a new, empty project folder, the equivalent starting command is:

```console
uv init --package --name eaas-llm-inference --python 3.13 llm_inference
```

This generates `pyproject.toml`, `.python-version`, `README.md` and `src/eaas_llm_inference/__init__.py` (and normally initializes Git). We replaced the generated greeting with our examples and their entry points, and retained support for Python 3.11 or newer. `uv.lock` and `.venv` are created later by locking and syncing. Do not rerun `uv init` inside this already initialized project.

The explicit `--package` flag makes the intended layout clear across uv versions. Before uv 0.12, plain `uv init` defaulted to a flat application; newer versions default to a packaged application. [uv project templates](https://docs.astral.sh/uv/concepts/projects/init/).

## Add the next experiment

Add another descriptively named `.py` module in `src/eaas_llm_inference/`, then run it from the project root:

```console
# After creating the file:
uv run python -m eaas_llm_inference.your_new_script
```

When a new exercise needs another library, use `uv add <package-name>`. That command updates the dependency declaration, lockfile and environment. Use `uv remove <package-name>` when it is no longer needed. To deliberately refresh the dependency versions, use `uv lock --upgrade` followed by `uv sync`, and review the resulting changes. [Managing a uv project](https://docs.astral.sh/uv/guides/projects/).

For a short terminal command, add a mapping under `[project.scripts]` and run `uv sync` to refresh the installed launchers. When experiments need shared implementation, add modules in the same package and import their functions. The initial examples remain small enough to teach directly.

## Check code quality

The project uses [Ruff](https://docs.astral.sh/ruff/) as linter and formatter. It is declared as a development dependency in `pyproject.toml`, so `uv sync` installs the locked version into `.venv` alongside the runtime libraries. Check every new or changed module from the project root:

```console
uv run ruff check
uv run ruff format
```

`ruff check` reports problems such as unused imports, functions without type hints or docstrings, an HTTP request without a timeout, or an `assert` used as a runtime check. `uv run ruff check --fix` applies the fixes Ruff considers safe. `ruff format` rewrites layout such as line breaks and spacing consistently without changing behavior; `uv run ruff format --check` only reports files it would change. Neither command runs the examples or sends a request.

The selected rule groups are listed under `[tool.ruff.lint]` in `pyproject.toml`, each with a short comment. Ruff checks only the Python files: code blocks in the Markdown guides are excerpts, some of them fragments of a call. [Ruff configuration](https://docs.astral.sh/ruff/configuration/), [rule reference](https://docs.astral.sh/ruff/rules/).

The examples annotate every function and variable with type hints. Ruff requires hints on functions but does not check whether hints are correct; a type checker does. This command checks them with mypy without adding it to the project, similar to the `--with debugpy` attach workflow:

```console
uv run --with mypy mypy --strict src
```

In VS Code, the recommended **Ruff** extension (`charliermarsh.ruff`) shows the same findings in the editor.

## Troubleshooting

| Observation | Next step |
|---|---|
| `uv` is not recognized | Install uv using the linked instructions, then reopen the terminal. |
| uv cannot find a project | Change into `llm_inference` before running these commands. |
| A library is missing when using plain `python` | Use `uv run ...` to select the project's environment. |
| The lockfile is out of date | After an intentional dependency edit, run `uv lock`, review the update, then sync. |
| `NVIDIA_API_KEY is missing from the local configuration.` | Check the private `.env` in the project root; the script stopped before HTTP. |
| NVIDIA returns an error | Use the [HTTP/API troubleshooting guide](nvidia_walkthrough.md#12-diagnose-the-layer-before-changing-the-prompt). |
| `Refusing to print or save: the capture contains the key.` | The response unexpectedly contained your key, so nothing was printed or saved. Inspect it privately in the [debugger](debugging.md#7-debug-the-live-request-safely). |

The offline preview uses a literal fake credential and does not read `.env`, so it can be projected in class. Keep the actual key and any live authorization headers off the shared screen.
