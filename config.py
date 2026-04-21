import os
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        return cleaned[1:-1]
    return cleaned

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Vertex AI configuration
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# OpenAI and OpenRouter configuration
OPENAI_BASE_URL = _clean_env_value(os.environ.get("OPENAI_BASE_URL"))
OPENROUTER_BASE_URL = _clean_env_value(os.environ.get("OPENROUTER_BASE_URL")) or "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL = _clean_env_value(os.environ.get("OPENROUTER_SITE_URL"))
OPENROUTER_APP_NAME = _clean_env_value(os.environ.get("OPENROUTER_APP_NAME"))

# Provider and model mapping configuration
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview"
]

PROVIDER_PREFIXES = ("openai/", "gemini/", "anthropic/", "openrouter/")


def strip_provider_prefix(model_name: str) -> str:
    for prefix in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def has_provider_prefix(model_name: str) -> bool:
    return model_name.startswith(PROVIDER_PREFIXES)


def resolve_model_name(original_model: str) -> Tuple[str, bool]:
    """Resolve request model to provider-prefixed model understood by LiteLLM."""
    clean_model = strip_provider_prefix(original_model)

    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean_model}", True

    if "haiku" in clean_model.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            return f"gemini/{SMALL_MODEL}", True
        if PREFERRED_PROVIDER == "openrouter":
            return f"openrouter/{SMALL_MODEL}", True
        return f"openai/{SMALL_MODEL}", True

    if "sonnet" in clean_model.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            return f"gemini/{BIG_MODEL}", True
        if PREFERRED_PROVIDER == "openrouter":
            return f"openrouter/{BIG_MODEL}", True
        return f"openai/{BIG_MODEL}", True

    if clean_model in GEMINI_MODELS and not original_model.startswith("gemini/"):
        return f"gemini/{clean_model}", True

    if clean_model in OPENAI_MODELS and not original_model.startswith("openai/"):
        return f"openai/{clean_model}", True

    if PREFERRED_PROVIDER == "openrouter" and not has_provider_prefix(original_model):
        return f"openrouter/{clean_model}", True

    return original_model, False
