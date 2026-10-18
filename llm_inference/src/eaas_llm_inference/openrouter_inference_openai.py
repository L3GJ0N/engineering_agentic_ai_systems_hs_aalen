"""The request of openrouter_inference.py, sent with the OpenAI SDK instead of Requests.

Setup: uv sync
Configuration is read from the project's private .env file at runtime.
Run from the project root: uv run openrouter-inference-openai
Compare line by line with openrouter_inference.py.
Read: ../../openrouter_walkthrough.md, section 12 (from the package directory)
"""

from pathlib import Path
from typing import Any

from dotenv import dotenv_values  # Read a local configuration file.
from openai import OpenAI  # Client for the OpenAI-compatible API format.
from openai.types.chat import ChatCompletion


def main() -> None:
    """Send one prompt to OpenRouter through the OpenAI SDK and print the answer."""
    # Same configuration as openrouter_inference.py.
    project_root: Path = Path(__file__).resolve().parents[2]
    env_path: Path = project_root / ".env"
    config: dict[str, str | None] = dotenv_values(env_path, interpolate=False)
    api_key: str | None = config.get("OPENROUTER_API_KEY")  # None if absent.
    if api_key is None or not api_key.strip():
        raise SystemExit("OPENROUTER_API_KEY is missing from the local configuration.")

    # "OpenAI" names the API format, not the destination: base_url sends every
    # request to OpenRouter. Nothing goes to OpenAI; no OpenAI account is involved.
    client: OpenAI = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        # Passed explicitly; otherwise the SDK would read OPENAI_API_KEY.
        api_key=api_key,
        # The SDK defaults are a 600 s read timeout and 2 automatic retries on
        # timeouts, connection errors, 408, 409, 429 and 5xx. Match the Requests
        # version instead: failed attempts count against the free daily limit.
        timeout=60,
        max_retries=0,
    )

    prompt: str = "Explain tokenization in one sentence."

    # The SDK builds the same HTTP request as openrouter_inference.py: POST to
    # base_url + "/chat/completions", the Authorization header and a JSON body.
    print("Sending request to OpenRouter; waiting for the response...")
    completion: ChatCompletion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        stream=False,  # Receive the complete response at once.
        # Fields outside OpenAI's own schema are merged into the JSON body.
        extra_body={"reasoning": {"enabled": False}},
    )
    # The status code stays hidden: an error status raises an exception instead,
    # such as openai.RateLimitError for 429, whose message includes OpenRouter's
    # explanation from the body.

    # Response fields outside OpenAI's schema are kept in model_extra: provider,
    # and an "error" that OpenRouter reports inside "200 OK", either instead of
    # choices or in choices[0] beside partial output.
    extra: dict[str, Any] = completion.model_extra or {}
    if "error" in extra:
        raise SystemExit(f"OpenRouter reported an error: {extra['error']}")
    choice_extra: dict[str, Any] = completion.choices[0].model_extra or {}
    if "error" in choice_extra:
        raise SystemExit(f"OpenRouter reported an error: {choice_extra['error']}")

    # The SDK has parsed the JSON into typed objects: attributes, not dict keys.
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
