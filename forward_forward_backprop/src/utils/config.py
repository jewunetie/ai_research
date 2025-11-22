"""Configuration management."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class Config:
    """Configuration object with dot notation access."""

    def __init__(self, config_source):
        """
        Initialize Config from file path or dictionary.

        Args:
            config_source: Either a file path (str/Path) or a dictionary
        """
        # Load config from file if path is provided
        if isinstance(config_source, (str, Path)):
            config_dict = load_config(config_source)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise TypeError(f"Config source must be str, Path, or dict, got {type(config_source)}")

        # Store the original dict
        self._config_dict = config_dict

        # Set attributes for dot notation access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        """
        Get config value using dot notation.

        Args:
            key: Key in dot notation (e.g., 'experiment.name')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self

        for k in keys:
            if isinstance(value, Config):
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            elif isinstance(value, dict):
                value = value.get(k, default)
                if value is default:
                    return default
            else:
                return default

        return value

    def __repr__(self):
        return f"Config({self._config_dict})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):  # Skip private attributes
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
