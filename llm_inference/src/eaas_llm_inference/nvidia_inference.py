"""One prompt, one NVIDIA inference request, one answer.

Setup: uv sync
Configuration is read from the project's private .env file at runtime.
Run from the project root: uv run nvidia-inference
Get a key: https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b
HTTP basics: ../../http_and_requests.md (from the package directory)
Walkthrough: ../../nvidia_walkthrough.md (from the package directory)
"""

from pathlib import Path

import requests  # General-purpose HTTP library, independent of NVIDIA.
from dotenv import dotenv_values  # Read a local configuration file.


def main():
    # In this editable course checkout, this file is in src/eaas_llm_inference/.
    # parents[0] is the package folder, [1] is src, [2] is the project root.
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    config = dotenv_values(env_path, interpolate=False)
    api_key = config.get("NVIDIA_API_KEY")
    if api_key is None or not api_key.strip():
        raise SystemExit("NVIDIA_API_KEY is missing from the local configuration.")

    prompt = "Explain tokenization in one sentence."

    # This call sends HTTPS to NVIDIA; the model runs on their servers.
    # Local Python waits here for the response. No model is loaded locally.
    print("Sending request to NVIDIA; waiting for the response...")
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},  # Send the selected value.
        json={
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": False,  # Receive the complete response at once.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
        allow_redirects=False,
    )
    print(f"Received HTTP {response.status_code} from NVIDIA.")

    # Back on the laptop: check, parse and display. These steps do no inference.
    response.raise_for_status()  # Raise for an HTTP error (4xx/5xx).
    result = response.json()  # Parse the received JSON into Python objects.
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
