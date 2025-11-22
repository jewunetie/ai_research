"""
Simple configuration loader (no dependencies required).

This is a lightweight version that doesn't require pydantic.
Use this for testing before installing all dependencies.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        # Default to config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def resolve_path(path: str | Path) -> Path:
    """
    Resolve path relative to project root.

    Args:
        path: Relative or absolute path

    Returns:
        Absolute Path object
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return get_project_root() / path
