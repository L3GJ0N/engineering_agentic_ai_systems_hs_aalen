"""Prepare and inspect a request locally. Nothing is sent; no key is needed.

Setup: uv sync
Run from the project root: uv run inspect-http-request
Read: ../../http_and_requests.md (from the package directory)
"""

import requests


def main():
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    # Keep this fake: the exercise prints the headers below.
    headers = {"Authorization": "Bearer CLASSROOM_PLACEHOLDER"}
    body = {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "messages": [{"role": "user", "content": "Explain tokenization in one sentence."}],
        "max_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Preparation serializes our data locally. It does not send a request.
    prepared = requests.Request(
        method="POST", url=url, headers=headers, json=body
    ).prepare()

    print("LOCAL PREVIEW ONLY: no request sent, no model executed.")
    print("Method:", prepared.method)
    print("URL:", prepared.url)
    print("Headers:", dict(prepared.headers))
    # Decode bytes as text for display; this is not LLM token decoding.
    print("Body:", prepared.body.decode("utf-8"))


if __name__ == "__main__":
    main()
