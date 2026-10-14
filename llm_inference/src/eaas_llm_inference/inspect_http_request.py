"""Prepare and inspect a request locally. Nothing is sent; no key is needed.

Setup: uv sync
Run from the project root: uv run inspect-http-request
Read: ../../http_and_requests.md (from the package directory)
"""

from typing import Any

import requests


def main() -> None:
    """Prepare the NVIDIA request locally and print its parts without sending it."""
    url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    # Keep this fake: the exercise prints the headers below.
    headers: dict[str, str] = {"Authorization": "Bearer CLASSROOM_PLACEHOLDER"}
    # The same body as the live starter. Its values mix strings, numbers,
    # booleans, lists and nested dictionaries, hence the type hint Any.
    body: dict[str, Any] = {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "messages": [
            {"role": "user", "content": "Explain tokenization in one sentence."}
        ],
        "max_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Preparation serializes our data locally. It does not send a request.
    # Request(...) only stores the arguments; prepare() encodes the body as
    # UTF-8 JSON bytes and adds the Content-Type and Content-Length headers.
    prepared: requests.PreparedRequest = requests.Request(
        method="POST", url=url, headers=headers, json=body
    ).prepare()

    print("LOCAL PREVIEW ONLY: no request sent, no model executed.")
    print("Method:", prepared.method)
    print("URL:", prepared.url)
    # requests.post(...) would also add User-Agent, Accept-Encoding, Accept and
    # Connection, plus Host while sending. Here we see what prepare() produced.
    print("Headers:", dict(prepared.headers))
    # A prepared body can in general be text, bytes or a file; json= gives bytes.
    if not isinstance(prepared.body, bytes):
        raise TypeError("Requests should have encoded the json= body as bytes.")
    # Decode bytes as text for display; this is not LLM token decoding.
    print("Body:", prepared.body.decode("utf-8"))


if __name__ == "__main__":
    main()
