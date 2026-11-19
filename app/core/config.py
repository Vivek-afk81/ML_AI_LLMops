from pathlib import Path

import yaml


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2] # parents[2] file location => app/core/config.py

# Configuration file path
CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"


def load_config() -> dict:
    """
    Load application configuration from YAML file.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) # convert yaml file into dictionary

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a valid YAML mapping."
        )

    return config


# Load configuration once when this module is imported
settings = load_config()
