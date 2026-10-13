# Before inference: HTTP and the Python Requests library

Two programs are involved: **our Python program asks NVIDIA's server to generate text; NVIDIA's server sends a response back.** They run on different computers, so our program must send a message over the network.

Work through this introduction alongside [the NVIDIA walkthrough](nvidia_walkthrough.md). Start with 20 minutes on the HTTP message and offline inspection; return to the configuration details while setting up and running the starter. The offline exercise needs no account, API key or GPU. The live [nvidia_inference.py](src/eaas_llm_inference/nvidia_inference.py) reads its key from a private `.env` file in the project root.

Use the standalone project in `llm_inference`. The [project README](README.md) explains installing uv and opening a terminal in this folder. All commands below run from this folder.

## 1. What is an HTTP request?

An **HTTP request** is a structured message from a client to a server. HTTP means *Hypertext Transfer Protocol*: shared rules for communicating. A **client** starts the exchange; a **server** receives requests and provides a service. These names describe software roles. Here, Python on your laptop is the client, and NVIDIA operates the server.

A browser uses HTTP to request a web page. Our script uses it to request computation. The response can contain structured data instead of a page. **HTTPS** carries HTTP over TLS, which protects the connection through encryption, integrity checks and server authentication. NVIDIA's service can read the request after decryption. [HTTP standard: clients, servers and HTTPS](https://www.rfc-editor.org/rfc/rfc9110.html).

```mermaid
sequenceDiagram
    participant P as Laptop: Python + Requests
    participant N as NVIDIA: API service
    participant M as NVIDIA: model runtime
    P->>P: Build a request locally
    P->>N: HTTPS request: please generate text
    Note over P: Waiting inside requests.post(...)
    N->>N: Check access and request data
    N->>M: Dispatch an accepted generation request
    M-->>N: Generated output
    N-->>P: HTTP response: status + headers + body
    P->>P: Read the response and print the answer
```

This shows responsibilities, not NVIDIA's exact internal topology. A rejected request can produce an error response before a model runs.

## 2. Read the address

Our URL is:

```text
https://integrate.api.nvidia.com/v1/chat/completions
```

| Part | Meaning in this exercise |
|---|---|
| `https://` | Use HTTP over a protected connection. |
| `integrate.api.nvidia.com` | Host name identifying the service destination. |
| `/v1/chat/completions` | Path identifying this service's chat-completion operation. |

An **endpoint** is an address through which a service exposes an operation. The HTTP method matters too: this operation expects **POST**, which submits data for processing. **GET** is commonly used to retrieve a resource. Pasting this URL into a browser's address bar does not reproduce our POST request, authorization header and JSON body.

The address identifies a service, not an individual GPU. The provider can route requests to different workers behind it. [NVIDIA endpoint reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b-infer).

## 3. The request has an envelope and a body

For this exercise, inspect four parts:

| Part | Concrete example | Purpose |
|---|---|---|
| Method | `POST` | Say what kind of HTTP operation we request. |
| Destination | URL above | Say which service operation to contact. |
| Headers | `Authorization`, `Content-Type` | Attach metadata such as credentials and body format. |
| Body | JSON containing `model` and `messages` | Supply the data the operation should process. |

**JSON** is a text format for structured data: objects, lists, strings, numbers, booleans and null. It lets programs exchange data without sharing a programming language.

This is the same JSON used in files ending in **`.json`**. A file stores JSON text on disk; an HTTP body can carry JSON text over the network without creating a file. In our starter, Requests serializes a Python dictionary directly into the body. `response.json()` performs the reverse kind of operation on the received body: parsing JSON into Python objects. [Python's JSON explanation](https://docs.python.org/3/library/json.html).

Here is a shortened HTTP/1.1-style illustration. It is **not a captured request or a complete message to send**; extra headers and body fields are omitted. The key is a fake classroom placeholder.

```http
POST /v1/chat/completions HTTP/1.1
Host: integrate.api.nvidia.com
Authorization: Bearer CLASSROOM_PLACEHOLDER
Content-Type: application/json

{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","messages":[{"role":"user","content":"Explain tokenization in one sentence."}],"stream":false}
```

The blank line separates headers from the body. `Content-Type` describes the body's format. `Bearer` identifies the authentication scheme; the following secret is the credential. Keep real keys out of printed requests and shared screenshots. [HTTP message anatomy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Messages).

Two separate headers make two separate statements: **`Authorization` supplies the credential; `Content-Type: application/json` identifies the body's format.** The headers are not themselves part of the JSON body. In Python we conveniently describe both headers and body using dictionaries, but Requests turns them into different parts of the HTTP message.

## 4. Is `requests` provided by NVIDIA?

**No. Requests is an independent, open-source Python library for HTTP communication.** It runs on your laptop and can communicate with many services. It contains neither this model's weights nor NVIDIA's inference engine. The project is maintained separately from NVIDIA. [Requests project documentation](https://requests.readthedocs.io/en/latest/).

We use **uv** to prepare this project's Python environment:

```powershell
uv sync
```

The project already declares `requests` and `python-dotenv`. These files have different responsibilities, as described in [uv's project guide](https://docs.astral.sh/uv/guides/projects/):

| File or folder | What it means |
|---|---|
| `pyproject.toml` | Declares the project, required libraries, supported Python versions (3.11 or newer), build backend and command entry points. |
| `uv.lock` | Records resolved dependency versions for reproducible installation. Share this file with the project. |
| `.python-version` | Selects Python 3.13 as the project's default interpreter version. |
| `.venv/` | Contains the local Python environment, installed dependencies and commands for our own package. uv creates it; each student has their own copy. |
| `src/eaas_llm_inference/` | Contains our Python package and its example modules. `__init__.py` marks the package. |

`uv sync` updates the lockfile if the project's requirements need it, installs the selected dependencies into `.venv`, and installs our own package in **editable mode**. Editable means that imports use the code in this checkout, so editing a module changes what the next run executes without copying it manually into `.venv`. uv prefers the versions already recorded in `uv.lock`; an ordinary sync does not upgrade packages just because newer releases exist. Downloading Python or packages can require internet access; this setup does not call NVIDIA's inference API. [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

The same automatic lockfile check applies to `uv run`. These defaults suit our interactive exercises. The optional `--locked` flag makes either command stop if the lockfile would need updating, which is useful in automated checks that require the committed lockfile to match the project.

Requests is an additional package, not part of Python's standard library. Installing it does not install Nemotron or create an NVIDIA account. [Requests installation](https://requests.readthedocs.io/en/latest/user/install/).

To execute our offline example in that environment:

```powershell
uv run inspect-http-request
```

uv selects the project environment, checks it is ready, and runs the installed `inspect-http-request` command. The `[project.scripts]` entry in `pyproject.toml` maps that name to `eaas_llm_inference.inspect_http_request:main`: import that module, then call its `main()` function. The command is a small launcher generated during installation; the HTTP example still consists of ordinary Python in the source module. Manual environment activation is unnecessary. [uv running project commands](https://docs.astral.sh/uv/guides/projects/#running-commands).

The module can also be run explicitly with `uv run python -m eaas_llm_inference.inspect_http_request`. Here, `-m` tells Python to execute the installed module; its `if __name__ == "__main__"` block calls `main()`. Both paths reach the same exercise code.

In the script:

```python
import requests
```

This loads the installed package so we can use its functions. **Importing it does not contact NVIDIA.** In `requests.post(...)`, the dot selects the package's `post` function and the parentheses call it.

Why does an independent library work? The client and server agree on two things:

| Agreement | Who defines it? | Example |
|---|---|---|
| Communication rules | HTTP standards | Methods, headers and status codes. |
| Application interface, or API contract | The service's API documentation | Accepted `model`, `messages`, settings and response fields. |

**API** means *Application Programming Interface*: rules for using functionality from software. This API is exposed over HTTP. Requests implements the general communication; our code supplies the NVIDIA-specific address and data. The server can be written in another language and does not need to use Requests.

Consequently, another HTTP client could send an equivalent request. However, changing the host alone does not make every provider compatible: authentication, supported models and accepted fields must match that provider's contract. HTTP itself assigns no meaning to `temperature` or `max_tokens`.

## 5. Map the Python arguments to the message

Read the actual call in [nvidia_inference.py](src/eaas_llm_inference/nvidia_inference.py):

| Python expression | What it controls |
|---|---|
| `requests.post(...)` | Construct and send an HTTP POST, then receive its response. |
| First argument, the URL | Where the request goes. |
| `headers={...}` | Header values to attach. |
| `json={...}` | Python data to serialize into the request body. Requests also supplies `Content-Type: application/json`. |
| `timeout=60` | Local connection/read waiting limits; not a total inference deadline. |
| `allow_redirects=False` | Whether the client follows a server's instruction to try another URL. Here, it does not. |

The last two settings are outside `json`: they configure the local client. They are not model settings. A client timeout does not prove remote work was cancelled. [Requests request arguments](https://requests.readthedocs.io/en/latest/api/#requests.request).

The `json` name is a function argument. With this argument, we give Requests a **Python dictionary**, not an already serialized JSON string. Requests converts it: for example, Python `False` becomes JSON `false`. [Requests JSON submission](https://requests.readthedocs.io/en/latest/user/quickstart/#more-complicated-post-requests).

There are three different representations to keep separate:

```text
LAPTOP                            NVIDIA SERVICE
Python dictionary
    -> JSON encoded as bytes  --> parse request data
                                  -> format conversation
                                  -> tokenize model input
                                  -> run model inference
```

Serializing JSON and encoding text as bytes do not perform LLM tokenization. The HTTP headers and whole JSON envelope are not automatically the prompt. Server-side software extracts the messages and prepares the model input.

### 5.1. Choose where the API key comes from

There are three decisions: **where to store the key, how Python reads it, and how the HTTP request carries it**. Changing the first two does not require changing NVIDIA's authorization header.

| Source and reader | What actually happens | Place in this course |
|---|---|---|
| Local `.env` + `dotenv_values(...)` | Read named values from a text file into a Python dictionary. | Our simple starter. |
| Process environment + `os.environ[...]` or `os.getenv(...)` | Read a value provided to the running process, for example through terminal/IDE configuration. | Compare later when deploying applications. |
| `.env` + `load_dotenv()`, then an environment lookup | Load file values into the process environment, then read from that environment. | A common alternative with an extra configuration layer. |
| JSON/INI file + `json.load(...)` / `configparser` | Parse another structured file format. | Alternative when an application already uses that format. |
| OS credential store + a library such as `keyring` | Retrieve a secret from the system's credential storage. | Later deployment/security discussion; additional setup depends on the backend. |

`os.getenv("NVIDIA_API_KEY")` is a different way of accessing the **same process environment** as `os.environ["NVIDIA_API_KEY"]`. The first returns `None` by default if missing; the second raises `KeyError`. It does not create a different place to store secrets. [Python environment access](https://docs.python.org/3/library/os.html#os.getenv).

`dotenv_values` returns parsed file values without populating the environment. `load_dotenv` instead loads values into the environment; by default it does not replace variables already set there. Our starter uses the first approach. [python-dotenv documentation](https://bbc2.github.io/python-dotenv/#load-configuration-without-altering-the-environment). Other readers have their own interfaces: [JSON files](https://docs.python.org/3/library/json.html), [INI files](https://docs.python.org/3/library/configparser.html), [credential stores](https://keyring.readthedocs.io/en/latest/).

### 5.2. Read our local configuration one line at a time

The file layout is:

```text
llm_inference/
    pyproject.toml        project and dependency declarations
    uv.lock               resolved dependency versions
    .python-version       default Python version
    .venv/                local Python environment created by uv
    .env.example          shareable template with an empty field
    .env                  your private configuration
    src/
        eaas_llm_inference/
            __init__.py
            nvidia_inference.py
            inspect_http_request.py
```

The current starter reads a private `.env` in the project root, beside `pyproject.toml`. [.env.example](.env.example) documents the expected configuration name with an empty field; it contains no credential:

```dotenv
NVIDIA_API_KEY=
```

For local setup, create the private `.env` from this empty template and configure `NVIDIA_API_KEY` there. If you already have private configuration in this project root, keep using it. Leave the tracked `.env.example` empty. The actual key belongs only in the private configuration, never in Python source or another committed file.

This is **plain `NAME=value` text**, not JSON or Python code. `.env` is a filename convention. An empty field is not a usable credential, so the script stops locally if that is all it finds. Creating the file alone does not change your terminal environment or send data anywhere; our explicit reader loads it. No Python literal should be replaced with a real key.

The earlier `uv sync` installs the reader too: `python-dotenv` is already declared in `pyproject.toml`. No separate installation command is needed.

The package's installation name is **`python-dotenv`**, while its Python import name is **`dotenv`**. Both refer to the same library. It is independent of NVIDIA. [Installation and import examples](https://bbc2.github.io/python-dotenv/).

The starter contains:

```python
from pathlib import Path
from dotenv import dotenv_values

project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
config = dotenv_values(env_path, interpolate=False)
api_key = config.get("NVIDIA_API_KEY")
```

| Expression | What happens locally |
|---|---|
| `__file__` | Identifies this module's file path: `src/eaas_llm_inference/nvidia_inference.py`. |
| `Path(__file__)` | Represents that path as a Python path object. |
| `.resolve()` | Resolves an absolute path to the module, including any symbolic links. |
| `.parents[2]` | Selects the project root in our checkout: parent 0 is `eaas_llm_inference`, parent 1 is `src`, parent 2 is `llm_inference`. |
| `project_root / ".env"` | Joins the root path with the filename `.env`. It neither creates nor reads the file. |
| `dotenv_values(env_path, ...)` | Opens the specified configuration file and parses its entries into a dictionary-like result. |
| `interpolate=False` | Keeps values literal instead of expanding expressions such as `${OTHER_VARIABLE}`. |
| `config.get("NVIDIA_API_KEY")` | Selects the value under our chosen configuration name; returns `None` if that name is absent. |
| The following `if` check | Stops before HTTP if the value is missing, empty or only whitespace. It does not verify the credential with NVIDIA. |

This explicitly locates the file relative to the module in our **editable course checkout**, rather than relative to the terminal's current directory. It climbs a fixed number of directories; it does not search for configuration files or use an environment override for the key. The exercise assumes this source layout, so this path is not a configuration strategy for a standalone wheel installed elsewhere. We can make deployment configuration explicit when we reach that topic. The `.env` file remains on disk between runs, and each new run reads it again. [Reader parameters](https://bbc2.github.io/python-dotenv/reference/#dotenv.dotenv_values).

`"NVIDIA_API_KEY"` in the source is a configuration name, not a place to insert a secret. `api_key` is the variable that receives the value at runtime. The source code contains these names and references; the actual key comes from the private configuration.

With this file-based reader, the real key exists as plain text in `.env` and as a string in Python memory. The folder's `.gitignore` excludes `.env` from ordinary Git tracking, but does not encrypt it or protect a manually shared ZIP/folder. Submit only the empty example template, never your personal configuration.

### 5.3. Turn the selected string into an HTTP header

Next, the starter constructs:

```python
headers={"Authorization": f"Bearer {api_key}"}
```

Read this from the inside out. `api_key` holds the string we selected. The `f` prefix tells Python to substitute that string at `{api_key}`. Python adds the literal word `Bearer` and a space before it. The resulting string becomes the value for the dictionary entry named `Authorization`. The `headers=` function argument gives that dictionary to Requests.

These names have different owners:

| Name | Who uses it? | Could we rename it? |
|---|---|---|
| `NVIDIA_API_KEY` | Our local configuration reader | Yes, if we change the file entry and Python lookup together. |
| `api_key` | Our Python program | Yes, if all references are changed together. |
| `Authorization` | HTTP authentication and NVIDIA's service | It must follow the API's authentication contract. |
| `Bearer` | The selected HTTP authentication scheme | It must match the scheme expected by the service. |

**Bearer** means that possession of the credential allows its holder to present it for access, subject to the service's checks. It does not encrypt the key; HTTPS protects the connection. The phrase *bearer token* refers to an authentication credential, not a vocabulary token produced by an LLM tokenizer. [Bearer authentication specification](https://www.rfc-editor.org/rfc/rfc6750.html).

This construction happens in laptop memory. It has not yet sent anything:

```text
LOCAL FILE             LOCAL PYTHON                  HTTP REQUEST HEADER
.env                   config dictionary
NVIDIA_API_KEY=...  ->  api_key string             ->  Authorization: Bearer ...
```

NVIDIA receives the selected secret in the header. Our code does not upload the `.env` file, its filename, the `config` dictionary or the variable name `api_key`. It sends the JSON body separately. The server validates access and processes the requested operation; the API key is not added to the conversation's `messages` by this script.

### 5.4. Identify the point where networking starts

The configuration reads above use local disk and memory. **Within the inference script**, networking starts when Python calls `requests.post(...)`: Requests prepares the HTTP message and uses its networking stack to contact the URL. Earlier uv setup may have downloaded Python or dependencies; that is a separate exchange with software download services. At a conceptual level, the inference destination host is resolved to a network address, a connection is established, and TLS protects the HTTP exchange. Connection details can be reused by clients; these are responsibilities, not a packet-by-packet trace.

On the remote side, NVIDIA handles authentication, permissions, request validation and scheduling. Accepted generation requests reach the model-serving system. A failed access check can return an error without running the LLM. The service then sends an HTTP response; our laptop checks the status, parses the JSON and prints the answer.

```mermaid
flowchart LR
    subgraph Laptop
        F[Private .env file] --> D[dotenv_values: config dictionary]
        D --> K[Select api_key string]
        K --> H[Authorization header]
        P[Prompt and settings] --> J[JSON body]
        H --> R[requests.post]
        J --> R
    end
    R -->|HTTPS request| A[NVIDIA API service]
    A -->|Accepted generation request| M[Remote model runtime]
```

Reading the file, building the dictionary and constructing the header do no inference. Passing the `headers` argument does not make Requests a model library. The remote API contract connects this HTTP request to model execution.

## 6. Inspect a request before sending it

Run [inspect_http_request.py](src/eaas_llm_inference/inspect_http_request.py) through its installed command:

```powershell
uv run inspect-http-request
```

It uses a literal fake credential and constructs the same body as the starter. Its key operation is:

```python
prepared = requests.Request(
    method="POST", url=url, headers=headers, json=body
).prepare()
```

**Preparing creates local data. This exercise sends nothing.** `Request` describes the message; `.prepare()` turns its fields into a form ready for sending. Our inspection script never calls `.post()` or `.send()`. Once Requests is installed, it works offline. Its printed components are not a packet capture: later sending can add further details, such as the host header. [Requests preparation](https://requests.readthedocs.io/en/latest/user/advanced/#prepared-requests).

In pairs:

1. Locate the method, host, path, credential header and prompt in the output.
2. Identify the JSON content type and the body length header generated by the library. `Content-Length` counts body bytes, not LLM tokens.
3. Change the prompt and rerun. Explain what changed without claiming that a model ran.
4. Find Python `False` in the source and JSON `false` in the printed body.
5. Predict whether `max_tokens` controls a local library or the remote service. Explain how its placement tells you.
6. Explain whether NVIDIA would notice if we renamed `NVIDIA_API_KEY` in both `.env` and the Python lookup. It would not: the transmitted header value would be unchanged. Explain why changing the header name itself is different.

This preview uses its own fake credential. It never reads your `.env`, so it is suitable for projecting and inspecting even after you configure the live starter. Do not replace the preview's placeholder with a real key.

Then return to the live starter. `requests.post(...)` combines preparing, sending and receiving into one call.

## 7. What comes back?

The variable `response` contains a local **Requests Response object**, with status, headers and body. It is not a model object.

| Expression | What you inspect |
|---|---|
| `response.status_code` | HTTP status number. |
| `response.headers` | Response metadata. |
| `response.text` | Body interpreted as text. |
| `response.json()` | Body parsed into Python objects, if valid JSON. |

These inspections do not start another inference request in our non-streaming starter. `response.raise_for_status()` raises an exception for 4xx/5xx responses. It does not validate the answer or check the JSON schema. [Requests response interface](https://requests.readthedocs.io/en/latest/api/#requests.Response).

A **200** response indicates successful HTTP handling; **401** indicates an authentication problem; **500** indicates a server error. A **202** can mean the request was accepted but work is not yet complete. Network failures can occur before any HTTP response is received. [HTTP status semantics](https://www.rfc-editor.org/rfc/rfc9110.html#name-status-codes).

Our starter expects NVIDIA's normal completed response and extracts:

```python
result = response.json()
answer = result["choices"][0]["message"]["content"]
```

`choices` is a list; `[0]` selects its first entry. `message` contains the assistant message; `content` contains its answer text. The API's data format determines this access path. A valid HTTP response can still contain an error, an incomplete job or a factually wrong answer. See [the walkthrough's response handling](nvidia_walkthrough.md#6-follow-the-python-execution) before extending the starter.

## 8. Explain the boundary before moving on

Ask each student to finish this sentence while pointing at the code:

> My laptop executes ______; it sends ______ to ______; the model executes ______; the response is parsed ______.

Expected explanation: Python and Requests run locally; they send an HTTPS request containing headers and a JSON body to NVIDIA's endpoint; the model runs remotely for an accepted generation request; Python parses the returned body locally. The wait inside `requests.post(...)` includes networking, service waiting and inference, so its duration is not GPU execution time alone.

Now continue with [account setup and the NVIDIA model page](nvidia_walkthrough.md).
