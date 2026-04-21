import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional


def _prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def _prompt_secret(text: str) -> str:
    # Keep plain input for broad Windows compatibility in packaged binaries.
    return input(f"{text}: ").strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anthropic Proxy CLI Setup")
    parser.add_argument(
        "--p",
        "--provider",
        dest="provider",
        choices=["openai", "google", "anthropic", "openrouter"],
        help="Preferred provider to use (openai, google, anthropic, openrouter)",
    )
    parser.add_argument("--big-model", dest="big_model", help="Set BIG_MODEL")
    parser.add_argument("--small-model", dest="small_model", help="Set SMALL_MODEL")
    parser.add_argument("--host", dest="host", help="Host to bind the server to")
    parser.add_argument("--port", dest="port", type=int, help="Port to bind the server to")
    return parser.parse_args()


def _has_real_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith("your-"):
        return False
    if stripped.startswith("sk-..."):
        return False
    return True


def _provider_has_credentials(provider: str, existing: Dict[str, str]) -> bool:
    if provider == "google":
        use_vertex = existing.get("USE_VERTEX_AUTH", "").strip().lower() == "true"
        if use_vertex:
            return _has_real_value(existing.get("VERTEX_PROJECT", "")) and _has_real_value(existing.get("VERTEX_LOCATION", ""))
        return _has_real_value(existing.get("GEMINI_API_KEY", ""))

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    key_name = key_map.get(provider)
    if not key_name:
        return False
    return _has_real_value(existing.get(key_name, ""))


def _choose_provider() -> str:
    providers = {
        "1": "openai",
        "2": "google",
        "3": "anthropic",
        "4": "openrouter",
    }

    print("\nChoose provider:")
    print("1) OpenAI")
    print("2) Google (Gemini)")
    print("3) Anthropic")
    print("4) OpenRouter")

    while True:
        choice = input("Selection [1-4]: ").strip()
        if choice in providers:
            return providers[choice]
        print("Invalid selection. Try again.")


def _upsert_env_file(path: Path, updates: Dict[str, str]) -> None:
    existing_lines = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    key_line_index: Dict[str, int] = {}
    key_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*$")

    for idx, line in enumerate(existing_lines):
        m = key_pattern.match(line)
        if m:
            key_line_index[m.group(1)] = idx

    for key, value in updates.items():
        escaped_value = value.replace('"', '\\"')
        rendered = f'{key}="{escaped_value}"'
        if key in key_line_index:
            existing_lines[key_line_index[key]] = rendered
        else:
            existing_lines.append(rendered)

    path.write_text("\n".join(existing_lines).rstrip() + "\n", encoding="utf-8")


def _build_env_updates(provider: str) -> Dict[str, str]:
    updates: Dict[str, str] = {"PREFERRED_PROVIDER": provider}

    if provider == "openai":
        updates["OPENAI_API_KEY"] = _prompt_secret("Paste OPENAI_API_KEY")

    elif provider == "google":
        vertex = _prompt("Use Vertex auth? (true/false)", "false").lower()
        updates["USE_VERTEX_AUTH"] = "true" if vertex == "true" else "false"

        if updates["USE_VERTEX_AUTH"] == "true":
            updates["VERTEX_PROJECT"] = _prompt("VERTEX_PROJECT")
            updates["VERTEX_LOCATION"] = _prompt("VERTEX_LOCATION", "us-central1")
        else:
            updates["GEMINI_API_KEY"] = _prompt_secret("Paste GEMINI_API_KEY")

        # OpenAI remains useful as fallback for models not in Gemini list.
        fallback = _prompt("Add OPENAI_API_KEY fallback now? (true/false)", "false").lower()
        if fallback == "true":
            updates["OPENAI_API_KEY"] = _prompt_secret("Paste OPENAI_API_KEY")

    elif provider == "anthropic":
        updates["ANTHROPIC_API_KEY"] = _prompt_secret("Paste ANTHROPIC_API_KEY")

    elif provider == "openrouter":
        updates["OPENROUTER_API_KEY"] = _prompt_secret("Paste OPENROUTER_API_KEY")
        updates["OPENROUTER_BASE_URL"] = _prompt("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        site_url = _prompt("OPENROUTER_SITE_URL (optional)", "")
        app_name = _prompt("OPENROUTER_APP_NAME (optional)", "")
        if site_url:
            updates["OPENROUTER_SITE_URL"] = site_url
        if app_name:
            updates["OPENROUTER_APP_NAME"] = app_name

    big_model = _prompt("BIG_MODEL (optional)", "")
    small_model = _prompt("SMALL_MODEL (optional)", "")
    if big_model:
        updates["BIG_MODEL"] = big_model
    if small_model:
        updates["SMALL_MODEL"] = small_model

    return updates


def _load_existing_env(path: Path) -> Dict[str, str]:
    """Read key=value pairs from an existing .env file."""
    existing: Dict[str, str] = {}
    if not path.exists():
        return existing
    key_pattern = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?([^"]*)"?\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        m = key_pattern.match(line)
        if m:
            existing[m.group(1)] = m.group(2)
    return existing


def main() -> None:
    print("Anthropic Proxy CLI Setup")
    args = _parse_args()

    env_path = Path(__file__).resolve().parent / ".env"
    existing = _load_existing_env(env_path)

    # Determine provider (CLI arg > existing .env > interactive choose)
    preferred = existing.get("PREFERRED_PROVIDER", "").lower()
    provider = args.provider or (preferred if preferred else "")
    if not provider:
        provider = _choose_provider()

    # If credentials exist for selected provider, keep keys and only gather model/provider overrides.
    if _provider_has_credentials(provider, existing):
        print(f"Using existing credentials for provider={provider}. Press Enter to keep defaults.")
        updates: Dict[str, str] = {"PREFERRED_PROVIDER": provider}

        current_big = existing.get("BIG_MODEL", "")
        current_small = existing.get("SMALL_MODEL", "")
        if args.big_model is not None:
            big_model = args.big_model
        else:
            big_model = _prompt("BIG_MODEL", current_big or None)

        if args.small_model is not None:
            small_model = args.small_model
        else:
            small_model = _prompt("SMALL_MODEL", current_small or None)

        if big_model:
            updates["BIG_MODEL"] = big_model
        if small_model:
            updates["SMALL_MODEL"] = small_model

        _upsert_env_file(env_path, updates)
    else:
        if args.provider:
            print(f"No configured credentials found for provider={provider}. Prompting for missing setup.")
        updates = _build_env_updates(provider)

        if args.big_model is not None:
            updates["BIG_MODEL"] = args.big_model
        if args.small_model is not None:
            updates["SMALL_MODEL"] = args.small_model

        _upsert_env_file(env_path, updates)

    for key, value in updates.items():
        os.environ[key] = value

    # Load .env into environment so server picks it up
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

    host = args.host or _prompt("Host", "0.0.0.0")
    if args.port is not None:
        port = args.port
    else:
        port_raw = _prompt("Port", "8082")
        try:
            port = int(port_raw)
        except ValueError:
            print(f"Invalid port '{port_raw}', defaulting to 8082")
            port = 8082

    print("\nStarting proxy server...")
    import uvicorn
    from server import app

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
