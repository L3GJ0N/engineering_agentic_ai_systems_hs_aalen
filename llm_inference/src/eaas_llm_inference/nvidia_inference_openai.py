"""The request of nvidia_inference.py, sent with the OpenAI SDK instead of Requests.

Setup: uv sync
Configuration is read from the project's private .env file at runtime.
Run from the project root: uv run nvidia-inference-openai
Compare line by line with nvidia_inference.py.
Read: ../../nvidia_walkthrough.md, section 14 (from the package directory)
"""

from pathlib import Path

from dotenv import dotenv_values  # Read a local configuration file.
from openai import OpenAI  # Client for the OpenAI-compatible API format.
from openai.types.chat import ChatCompletion


def main() -> None:
    """Send one prompt to NVIDIA through the OpenAI SDK and print the answer."""
    # Same configuration as nvidia_inference.py.
    project_root: Path = Path(__file__).resolve().parents[2]
    env_path: Path = project_root / ".env"
    config: dict[str, str | None] = dotenv_values(env_path, interpolate=False)
    api_key: str | None = config.get("NVIDIA_API_KEY")  # None if the name is absent.
    if api_key is None or not api_key.strip():
        raise SystemExit("NVIDIA_API_KEY is missing from the local configuration.")

    # "OpenAI" names the API format, not the destination: base_url sends every
    # request to NVIDIA. Nothing goes to OpenAI; no OpenAI account is involved.
    client: OpenAI = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        # Passed explicitly; otherwise the SDK would read OPENAI_API_KEY.
        api_key=api_key,
        # The SDK defaults are a 600 s read timeout and 2 automatic retries on
        # timeouts, connection errors, 408, 409, 429 and 5xx. Match the Requests
        # starter instead: on the free tier, every retry counts against the limit.
        timeout=60,
        max_retries=0,
    )

    prompt: str = "Explain tokenization in one sentence."

    # The SDK builds the same HTTP request as the Requests starter: POST to
    # base_url + "/chat/completions", the Authorization header and a JSON body.
    print("Sending request to NVIDIA; waiting for the response...")
    completion: ChatCompletion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        stream=False,  # Receive the complete response at once.
        # Fields outside OpenAI's own schema are merged into the JSON body.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    # The status code stays hidden: an error status raises an exception instead,
    # such as openai.RateLimitError for 429, whose message includes NVIDIA's
    # explanation from the body.

    # The SDK has parsed the JSON into typed objects: attributes, not dict keys.
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
