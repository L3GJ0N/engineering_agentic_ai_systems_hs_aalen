"""Send one NVIDIA request, then print and save a key-free capture of it.

Setup: uv sync
Configuration is read from the project's private .env file at runtime.
Run from the project root: uv run capture-nvidia-response
Each run sends one live request and writes outputs/nvidia_<time>_capture.json.
Read: ../../http_and_requests.md, section 7 (from the package directory)
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

# Header values that can carry credentials. They are replaced before printing
# or saving, so a capture can be shown in class or shared.
SENSITIVE_HEADERS: frozenset[str] = frozenset({"authorization", "cookie", "set-cookie"})


def without_secrets(headers: Mapping[str, str | bytes]) -> dict[str, str]:
    """Copy headers as text and replace the values that can carry credentials."""
    safe: dict[str, str] = {}
    for name, value in headers.items():
        # Requests also accepts bytes; HTTP header bytes are read as Latin-1.
        text: str = value.decode("latin-1") if isinstance(value, bytes) else value
        safe[name] = "<removed>" if name.lower() in SENSITIVE_HEADERS else text
    return safe


def main() -> None:
    """Send one prompt to NVIDIA, then print and save the whole exchange."""
    # Same configuration and request as nvidia_inference.py.
    project_root: Path = Path(__file__).resolve().parents[2]
    config: dict[str, str | None] = dotenv_values(
        project_root / ".env", interpolate=False
    )
    api_key: str | None = config.get("NVIDIA_API_KEY")
    if api_key is None or not api_key.strip():
        raise SystemExit("NVIDIA_API_KEY is missing from the local configuration.")

    print("Sending request to NVIDIA; waiting for the response...")
    response: requests.Response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "messages": [
                {"role": "user", "content": "Explain tokenization in one sentence."}
            ],
            "max_tokens": 256,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
        allow_redirects=False,
    )
    # From here on, everything runs locally on the received response.

    # response.request is what was actually sent, including the default headers
    # Requests added. Its json= body is bytes; parse it back for readability.
    sent: requests.PreparedRequest = response.request
    sent_body: Any = json.loads(sent.body) if isinstance(sent.body, bytes) else None

    # An error page need not be JSON; keep its text instead.
    try:
        received_body: Any = response.json()
    except requests.JSONDecodeError:
        received_body = response.text

    now: datetime = datetime.now(UTC)
    capture: dict[str, Any] = {
        "captured_at": now.isoformat(timespec="seconds"),
        "request": {  # Built on this laptop.
            "method": sent.method,
            "url": sent.url,
            "headers": without_secrets(sent.headers),
            "body": sent_body,
        },
        "response": {
            # Status line, headers and body come from NVIDIA's service.
            "status_code": response.status_code,
            "reason": response.reason,
            "headers": without_secrets(response.headers),
            "body_bytes": len(response.content),  # After any decompression.
            "body": received_body,
            # Measured on this laptop: from sending until the headers arrived.
            "elapsed_seconds": response.elapsed.total_seconds(),
        },
    }
    capture_json: str = json.dumps(capture, indent=2, ensure_ascii=False)
    # Last safety net: never print or save anything that contains the key.
    if api_key.strip() in capture_json:
        raise SystemExit("Refusing to print or save: the capture contains the key.")
    print(capture_json)

    output_dir: Path = project_root / "outputs"  # Ignored by Git.
    output_dir.mkdir(exist_ok=True)
    path: Path = output_dir / f"nvidia_{now:%Y%m%dT%H%M%SZ}_capture.json"
    # Mode "x" creates a new file and never overwrites an earlier capture.
    with path.open("x", encoding="utf-8") as file:
        file.write(capture_json + "\n")
    print(f"Saved key-free capture: {path.relative_to(project_root)}")
    if not response.ok:
        print(f"HTTP {response.status_code}: the response body above explains why.")


if __name__ == "__main__":
    main()
