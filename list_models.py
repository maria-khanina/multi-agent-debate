import os
from typing import Any

from debate_settings import GEMINI_MODELS_URL, REQUEST_TIMEOUT_SECONDS
import requests


def _request_models(api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        GEMINI_MODELS_URL,
        params={"key": api_key},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("models", [])


def list_available_models() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set.")
        return
    models = _request_models(api_key)
    print("---Available Models---")
    for model in models:
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"])


if __name__ == "__main__":
    list_available_models()
