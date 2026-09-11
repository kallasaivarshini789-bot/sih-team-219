"""
Configuration Parser Module
Loads YAML configuration files into accessible python dictionary or dot-notation object.
"""

from pathlib import Path
import yaml
from typing import Dict, Any


class ConfigDict(dict):
    """Dictionary supporting dot-notation attribute access."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return ConfigDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str = "configs/config.yaml") -> ConfigDict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ConfigDict object containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        # Fallback to root relative if executed from subdirs
        root_path = Path(__file__).resolve().parent.parent / config_path
        if root_path.is_file():
            path = root_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    return ConfigDict(raw_cfg)
