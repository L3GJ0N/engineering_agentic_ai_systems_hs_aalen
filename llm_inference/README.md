# LLM inference examples

This standalone **packaged uv project** starts with one NVIDIA inference request and an offline HTTP inspection. It follows a `uv init --package` scaffold, with Python code in `src/eaas_llm_inference/`. We will add modules for other providers, tokenization, reasoning and evaluation here as the course develops. All examples share one declared set of dependencies and one local Python environment.

Start with [HTTP, configuration and Requests](http_and_requests.md), then follow [NVIDIA account setup and inference](nvidia_walkthrough.md). These guides and the two scripts are self-contained; provider comparisons and tokenizer viewers will be added as later examples.

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

`uv sync` prepares the local `.venv` with the resolved dependencies and installs this project in **editable mode**: imports use the Python files in `src/`, so edits to their code take effect on the next run. It also installs the two command entry points declared in `pyproject.toml`. `uv run` prepares the environment as needed and runs a command using it. No manual environment activation is needed. These commands update `uv.lock` when project requirements require it; a new package release alone does not cause an upgrade. Add `--locked` when an outdated or missing lockfile should be an error, for example in CI. Initial setup needs network access when Python or packages are not already available. The inspection script itself sends no network request and needs no key. [uv environment and lock behavior](https://docs.astral.sh/uv/concepts/projects/sync/).

The Python code retrieves the credential from your private **`.env`** in the project root, beside `pyproject.toml`. The tracked [.env.example](.env.example) declares only the configuration field, with an empty value:

```dotenv
NVIDIA_API_KEY=
```

Configure the value only in your private, untracked `.env`. Python source uses the configuration field's name and the value read at runtime; it contains no key literal or paste-here placeholder. If you already configured `.env`, keep using it. In Windows, check that the filename is `.env`, not `.env.txt`. Then run:

```console
uv run nvidia-inference
```

The program reads the file, sends one request to NVIDIA, and prints the HTTP status and generated answer for a normal completed response. The model runs remotely; no GPU or model download is needed on your laptop. The current prompt asks for a one-sentence explanation of tokenization. Change that string in the script for your next experiment.

Keep `.env` in this project root. The code explicitly resolves that root from its location in `src/eaas_llm_inference/`; see [the configuration explanation](http_and_requests.md#52-read-our-local-configuration-one-line-at-a-time). Neither command needs a key in a command-line argument, and the live starter does not use uv's `--env-file` option: its Python code explicitly reads `.env` with `dotenv_values`.

## Understand the project files

```text
llm_inference/
    pyproject.toml           project metadata and dependency requirements
    uv.lock                  resolved package versions and distributions
    .python-version          preferred Python version: 3.13
    .gitignore               files excluded from ordinary Git tracking
    .env.example             configuration field with an empty value
    http_and_requests.md     the underlying mechanisms
    nvidia_walkthrough.md    account setup and NVIDIA details
    README.md                this guide
    .venv/                   local Python environment created by uv
    .env                     private key, created by you
    .vscode/                 debugger, uv task and extension recommendations
    src/
        eaas_llm_inference/
            __init__.py              marks the Python package
            nvidia_inference.py      one live inference request
            inspect_http_request.py  inspect a fake request without sending it
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

For example, `nvidia-inference = "eaas_llm_inference.nvidia_inference:main"` means "import this module and call its `main` function." The launcher is created during installation. `uv run` selects the environment and runs that launcher. NVIDIA inference starts only when the function reaches `requests.post(...)` and NVIDIA accepts the request.

You can also run the modules explicitly. Python's `-m` option finds an installed module and executes it; the `if __name__ == "__main__"` block then calls `main()`:

```console
uv run python -m eaas_llm_inference.inspect_http_request
uv run python -m eaas_llm_inference.nvidia_inference
```

The key-loading path is intentionally tied to this course checkout and uv's editable installation. Keep `.env` in the project root. A separately distributed wheel would need its own configuration-location policy.

## Debug in VS Code

uv manages Python, packages and the project environment. VS Code's **Python Debugger** extension controls execution through `debugpy`. The model still runs at NVIDIA; the debugger can follow the local client through request construction, sending, response parsing and printing. It cannot step through the remote model's forward pass.

| Option | How execution starts | When to use it |
|---|---|---|
| Start Debugging with the supplied launch configurations | A `uv sync` task prepares `.venv`; VS Code launches that environment's Python with a module. | Recommended for the course's first debugger exercise. |
| Debug the currently open Python file | After `uv sync`, select `.venv` and choose **Python Debugger: Debug Python File**. | Quick exploration; the selected interpreter and active file matter. |
| Start with `uv run`, then attach | `uv run --with debugpy` starts Python and waits; VS Code connects to it. | Learn the distinction between launching and attaching, or retain a terminal-driven workflow. |

### Launch using the uv environment

Use **File > Open Folder** and select **`llm_inference` itself**, or keep the outer **`engineering_agentic_ai_systems_hs_aalen` repository folder** open. We provide configurations at both folder roots: the project-local [launch.json](.vscode/launch.json) uses local paths, while the repository-root [launch.json](../.vscode/launch.json) points into this project. VS Code reads the opened folder's configuration; it does not recursively discover every nested `.vscode` folder. Both sets show the same **LLM:** configuration names. The steps below use menu and button names; keyboard shortcuts are optional and listed separately for each platform. Labels use VS Code's English interface.

Install Microsoft's **Python** (`ms-python.python`) and **Python Debugger** (`ms-python.debugpy`) extensions if needed; the workspace recommends them. No Copilot subscription is required. Open **Terminal > New Terminal** in VS Code. The shell commands below assume the terminal is in `llm_inference`; if it starts at the outer `engineering_agentic_ai_systems_hs_aalen` root, first run `cd llm_inference`. Then prepare the environment:

```console
uv sync
```

The same `uv sync` and `uv run ...` commands work in Windows PowerShell, macOS shells such as zsh, and Ubuntu/Linux shells such as bash. No manual virtual-environment activation is needed. Each student creates `.venv` locally with uv; do not copy it between operating systems.

| Where Python runs | Interpreter selected by the supplied launch configurations |
|---|---|
| Windows | `.venv/Scripts/python.exe` |
| Ubuntu / other Linux distributions | `.venv/bin/python` |
| macOS, including Intel and Apple silicon | `.venv/bin/python` |

The Windows override in `launch.json` selects the first path; macOS and Linux use the default path. These paths are relative to each student's project folder. The interpreter's actual platform and architecture come from their local uv setup.

If using **WSL**, open the folder in a VS Code WSL window and run uv inside that Linux environment. Install the Python extensions in WSL when prompted and use a Linux `.venv` there. A separate checkout inside WSL avoids mixing Windows and Linux environments. [VS Code with WSL](https://code.visualstudio.com/docs/remote/wsl).

Opening from a terminal is an optional alternative: `code -n .` opens the current project in a new VS Code window when `code` is on PATH. On macOS, use **View > Command Palette**, run **Shell Command: Install 'code' command in PATH**, and restart the terminal if you want this shortcut. **File > Open Folder** works without configuring the shell command. [VS Code on macOS](https://code.visualstudio.com/docs/setup/mac).

1. Open **View > Command Palette**, search for **Python: Select Interpreter**, and select this project's `.venv`. Use the platform's interpreter path from the table above if you need to browse for it; from the outer repository root it is under `llm_inference/`. The project-local `python.defaultInterpreterPath` setting helps a newly opened project workspace; explicitly select the interpreter if VS Code previously remembered a different one. The supplied named launch configurations also specify this interpreter directly.
2. Open [inspect_http_request.py](src/eaas_llm_inference/inspect_http_request.py). Click the editor gutter beside `prepared = requests.Request(...)` to set a breakpoint. A red dot marks it.
3. Select the **Run and Debug** icon in the Activity Bar: the play triangle with a small bug. Choose **LLM: Inspect HTTP (offline)** in the configuration dropdown, then click the triangle beside it to start debugging. The **Run > Start Debugging** menu action starts the selected configuration too.
4. The uv task prepares the environment (`uv: sync` in the project folder, or `uv: sync llm_inference` in the outer repository folder). The debugger then pauses at the module's entry because `stopOnEntry` is enabled. Click **Continue** in the debug toolbar to reach your breakpoint; hovering over a toolbar button shows its name.
5. Click **Step Over** to execute the preparation statement. Inspect `prepared.method`, `prepared.url` and `prepared.body` in **Watch** or the **Debug Console**. This script only prepares a fake request; it sends nothing and reads no credential.
6. Open [nvidia_inference.py](src/eaas_llm_inference/nvidia_inference.py). Set breakpoints on `response = requests.post(...)`, on the following HTTP-status print, and on the final `print(result[...])`. Select **LLM: NVIDIA inference (live)** and click **Start Debugging**.

A breakpoint pauses **before** its line executes. At the first request breakpoint, `response` does not exist yet. Choosing **Step Over** executes the HTTP call and waits for the complete response. At the following print, inspect `response.status_code`, `response.headers.get("Content-Type")` and `response.text`. At the final print, `result` has been assigned, so inspect `result["choices"][0]["message"]["content"]`.

The live debugger can display secrets from memory: `api_key`, the `config` dictionary and `response.request.headers` can expose the credential. Use the offline configuration for screen-shared demonstrations and inspect live responses privately.

Default shortcuts are optional conveniences:

| Action | Menu or mouse route | Windows | Ubuntu / Linux | macOS |
|---|---|---|---|---|
| Open Run and Debug | Select its Activity Bar icon. | Ctrl+Shift+D | Ctrl+Shift+D | Cmd+Shift+D |
| Open Command Palette | View > Command Palette | Ctrl+Shift+P | Ctrl+Shift+P | Cmd+Shift+P |
| Toggle a breakpoint | Click the editor gutter beside the line. | F9 | F9 | F9 |
| Start debugging | Run > Start Debugging | F5 | F5 | F5 |
| Continue to the next stop | Continue button in the debug toolbar | F5 | F5 | F5 |
| Step over a statement | Step Over button in the debug toolbar | F10 | F10 | F10 |
| Step into a function | Step Into button in the debug toolbar | F11 | F11 | F11 |
| Step out of a function | Step Out button in the debug toolbar | Shift+F11 | Shift+F11 | Shift+F11 |
| Stop a launched session | Stop button in the debug toolbar | Shift+F5 | Shift+F5 | Shift+F5 |

Here **Cmd** means the macOS Command key (`⌘`). Depending on the keyboard and OS settings, function keys may require **Fn** or may be intercepted by system actions. Keymap extensions and personal shortcuts can also change the defaults. Use the menu/button route whenever a shortcut does something else. To inspect the actual binding, open **View > Command Palette**, choose **Preferences: Open Keyboard Shortcuts**, and search for the action. [VS Code debugging defaults](https://code.visualstudio.com/docs/debugtest/debugging), [keyboard shortcut configuration](https://code.visualstudio.com/docs/configure/keybindings).

If the Run and Debug icon is hidden, **View > Command Palette > Debug: Select and Start Debugging** lets you choose one of the supplied configurations directly. [Selecting a debug configuration](https://code.visualstudio.com/docs/debugtest/debugging-configuration).

Both launch configurations set `justMyCode: true`. This focuses stepping on our code. Set it to `false` for a later exercise inside Requests; stepping into that library still runs locally. [Microsoft's Python debugging guide](https://code.visualstudio.com/docs/python/debugging).

### Read the debugger configuration

The files under this project's [.vscode](.vscode/launch.json) describe the editor workflow. The repository-root [configuration](../.vscode/launch.json) provides the same launches with paths prefixed by `llm_inference/` and a project-specific task name:

| File or setting | Role |
|---|---|
| [launch.json](.vscode/launch.json) | Defines the offline and live module launches, plus the attach configuration. |
| `"type": "debugpy"` | Uses Microsoft's Python debugger. |
| `"request": "launch"` | Starts a new Python process under the debugger. |
| `"module": "eaas_llm_inference.nvidia_inference"` | Equivalent module target to `python -m ...`; the module's `__main__` guard calls `main()`. |
| `"python"` and its `"windows"` override | Explicitly select `.venv`'s interpreter on each platform. uv itself is not a Python interpreter. |
| `"preLaunchTask": "uv: sync"` | Runs the task declared in [tasks.json](.vscode/tasks.json) before starting Python. That task executes ordinary `uv sync` in the project root. |
| `"cwd": "${workspaceFolder}"` | Makes the opened `llm_inference` folder the working directory. |
| `"envFile": ""` | Disables the extension's automatic `.env` loading. Our Python code still reads its private configuration explicitly with `dotenv_values`. |
| `"stopOnEntry": true` | Pauses at module entry, even before you set a breakpoint. |
| [settings.json](.vscode/settings.json) | Suggests the uv environment to the Python extension. |
| [extensions.json](.vscode/extensions.json) | Recommends the two Microsoft extensions. |

The launch route is `uv sync -> .venv -> VS Code/debugpy -> Python module -> main()`. It uses uv's environment, but does not literally run the `uv run nvidia-inference` shell command. The extension provides its debugger, so this route does not require adding `debugpy` to the project's dependencies. `envFile` behavior is implemented in the extension's [launch resolver](https://github.com/microsoft/vscode-python-debugger/blob/main/src/extension/debugger/configuration/resolvers/launch.ts) and [environment-file reader](https://github.com/microsoft/vscode-python-debugger/blob/main/src/extension/common/variables/environment.ts).

### Attach to a process started by uv

First set a breakpoint in the offline example. Use the terminal inside the same VS Code environment as the debugger (for WSL, use its WSL terminal). Then, from the project root, run:

```console
uv run --with debugpy python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m eaas_llm_inference.inspect_http_request
```

The terminal appears to wait: that is expected. In **Run and Debug**, select **LLM: Attach to uv/debugpy (5678)** and click **Start Debugging**. The process can now proceed to your breakpoint. Finish this session before starting another on the same port.

For live inference, use the same workflow with the live module:

```console
uv run --with debugpy python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m eaas_llm_inference.nvidia_inference
```

`--with debugpy` makes the debugger available for this invocation without editing `pyproject.toml` or `uv.lock`; its first use may download the package. `python -m debugpy` starts the debugger, while the second `-m` selects our application module. `--listen` exposes a debugger endpoint on this machine's loopback address, and `--wait-for-client` waits until a debugging client attaches before executing the target. This debug port is separate from the HTTPS connection to NVIDIA. [uv additional dependencies](https://docs.astral.sh/uv/concepts/projects/run/#requesting-additional-dependencies), [debugpy command-line usage](https://github.com/microsoft/debugpy#command-line-usage).

For a persistent development dependency instead, use `uv add --dev debugpy` once and omit `--with debugpy` from future commands. That variant records the debugger version in the project's lockfile. The project currently uses the temporary `--with` variant for attach exercises.

If an attached process remains active after disconnecting, focus its terminal and use **Ctrl+C** to interrupt it (Control, including on macOS), or terminate that terminal with its trash-can button. Restart the terminal command for the next attach session.

### Debugger troubleshooting

| Observation | Check |
|---|---|
| The supplied configurations are missing, or VS Code asks to create `launch.json` | Open either `llm_inference` or the outer `engineering_agentic_ai_systems_hs_aalen` folder, where configurations are supplied. Close the debugger-selection popup and try **View > Command Palette > Developer: Reload Window** if these files have just been added. An unrelated parent folder needs its own configuration. |
| A documented shortcut does nothing or triggers a system action | Use the named menu or toolbar button; inspect the actual binding with **Preferences: Open Keyboard Shortcuts**. |
| `code` is not recognized in the terminal | Open the folder from VS Code's **File > Open Folder** menu. The command-line launcher is optional. |
| Python debugging is unavailable | Install/enable Microsoft's Python and Python Debugger extensions. |
| The `uv: sync` task cannot find uv | Run `uv --version` in VS Code's terminal; restart VS Code after installing uv so it gets the updated PATH. |
| A current-file launch uses the wrong Python | Run **Python: Select Interpreter** and choose this project's `.venv`. |
| `No module named eaas_llm_inference` | Run `uv sync` in the project folder and check the chosen interpreter. |
| An attach launch cannot connect | Start the terminal command first; use the same port in both places. |
| Port 5678 is occupied | End the earlier debugger process or select another free port in both the terminal command and attach configuration. |
| A breakpoint is hollow or never reached | Check that the selected configuration targets the same module you are editing and that execution reaches the line. |

## How the scaffold was chosen

The project started from a scaffold generated with **uv 0.8.19**, then added the examples, dependency requirements and guides. For a new, empty project folder, the equivalent starting command is:

```console
uv init --package --name eaas-llm-inference --python 3.13 llm_inference
```

This generates `pyproject.toml`, `.python-version`, `README.md` and `src/eaas_llm_inference/__init__.py` (and normally initializes Git). We replaced the generated greeting with our two examples and their entry points, and retained support for Python 3.11 or newer. `uv.lock` and `.venv` are created later by locking and syncing. Do not rerun `uv init` inside this already initialized project.

The explicit `--package` flag makes the intended layout clear across uv versions. Before uv 0.12, plain `uv init` defaulted to a flat application; newer versions default to a packaged application. [uv project templates](https://docs.astral.sh/uv/concepts/projects/init/).

## Add the next experiment

Add another descriptively named `.py` module in `src/eaas_llm_inference/`, then run it from the project root:

```console
# After creating the file:
uv run python -m eaas_llm_inference.your_new_script
```

When a new exercise needs another library, use `uv add <package-name>`. That command updates the dependency declaration, lockfile and environment. Use `uv remove <package-name>` when it is no longer needed. To deliberately refresh the dependency versions, use `uv lock --upgrade` followed by `uv sync`, and review the resulting changes. [Managing a uv project](https://docs.astral.sh/uv/guides/projects/).

For a short terminal command, add a mapping under `[project.scripts]` and run `uv sync` to refresh the installed launchers. When experiments need shared implementation, add modules in the same package and import their functions. The initial examples remain small enough to teach directly.

## Troubleshooting

| Observation | Next step |
|---|---|
| `uv` is not recognized | Install uv using the linked instructions, then reopen the terminal. |
| uv cannot find a project | Change into `llm_inference` before running these commands. |
| A library is missing when using plain `python` | Use `uv run ...` to select the project's environment. |
| The lockfile is out of date | After an intentional dependency edit, run `uv lock`, review the update, then sync. |
| `NVIDIA_API_KEY is missing from the local configuration.` | Check the private `.env` in the project root; the script stopped before HTTP. |
| NVIDIA returns an error | Use the [HTTP/API troubleshooting guide](nvidia_walkthrough.md#12-diagnose-the-layer-before-changing-the-prompt). |

The offline preview uses a literal fake credential and does not read `.env`, so it can be projected in class. Keep the actual key and any live authorization headers off the shared screen.
