"""Configuration management for the ReasoningBank library."""

import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration settings.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
