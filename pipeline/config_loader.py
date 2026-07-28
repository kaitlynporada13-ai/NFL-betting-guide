"""
Configuration loader utility.
Loads settings.yaml and resolves environment variables.
"""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


# Load .env file from config directory
_config_dir = Path(__file__).parent.parent / "config"
_env_path = _config_dir / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _resolve_env_vars(value):
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    if isinstance(value, str):
        pattern = r"\$\{(\w+)\}"
        matches = re.findall(pattern, value)
        for match in matches:
            env_val = os.environ.get(match, "")
            value = value.replace(f"${{{match}}}", env_val)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_settings() -> dict:
    """Load and return the settings.yaml config with env vars resolved."""
    settings_path = _config_dir / "settings.yaml"
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    return _resolve_env_vars(settings)


def load_stadiums() -> dict:
    """Load stadium database from stadiums.yaml."""
    stadiums_path = _config_dir / "stadiums.yaml"
    with open(stadiums_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir(subdir: str = "") -> Path:
    """Return path to a data subdirectory, creating it if needed."""
    data_dir = get_project_root() / "data" / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
