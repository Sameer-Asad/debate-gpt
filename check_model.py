"""Check which OpenRouter models work with YOUR key.

Run from the repo root (venv active):
    python check_models.py

It sends one tiny request to each model and prints OK or FAILED.
Add or remove model IDs in CANDIDATES as you like.
"""
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")
if not key:
    sys.exit("OPENROUTER_API_KEY is missing in your .env file")

CANDIDATES = [
    "google/gemma-3-12b-it:free",
    "google/gemma-3-12b-it",
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-nemo",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-14b",
]


def clean(text: str) -> str:
    return text.encode("ascii", "ignore").decode().replace("\n", " ").strip()


for model in CANDIDATES:
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=300,
            timeout=60,
            max_retries=0,
        )
        reply = llm.invoke("Write one short sentence about debating.")
        print(f"OK      {model}: {clean(str(reply.content))[:80]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED  {model}: {clean(str(exc))[:110]}")