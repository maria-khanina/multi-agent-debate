import os
import requests


def list_available_models() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set.")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    models = data.get("models", [])
    
    print("---Available Models---")
    for model in models:
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"])


if __name__ == "__main__":
    list_available_models()