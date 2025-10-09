import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("ANTHROPIC_API_KEY")
url = "https://api.anthropic.com/v1/messages"

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

payload = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 50,
    "messages": [
        {"role": "user", "content": "Give me a sentence of 5 words"}
    ],
    "temperature": 0.7,
    "top_p": 1.0,
    "system": "You are a helpful AI Assistant"
}

print("=== REQUEST DEBUG ===")
print("URL:", url)
print("Headers:", json.dumps({k: v if k != "x-api-key" else "***" for k, v in headers.items()}, indent=2))
print("Payload:", json.dumps(payload, indent=2))
print("===================\n")

response = requests.post(url, headers=headers, json=payload)
print("Status Code:", response.status_code)
print("Response:", response.text)
