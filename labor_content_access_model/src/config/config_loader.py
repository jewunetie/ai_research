"""Configuration management for the simulation"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file. If None, uses default config

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has all required fields

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['simulation', 'agents', 'marketplace', 'revenue']

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    # Validate simulation section
    sim_required = ['random_seed', 'num_steps']
    sim_missing = [key for key in sim_required if key not in config['simulation']]
    if sim_missing:
        raise ValueError(f"Missing required simulation config keys: {sim_missing}")

    # Validate agents section
    if 'users' not in config['agents']:
        raise ValueError("Missing 'users' in agents config")
    if 'creators' not in config['agents']:
        raise ValueError("Missing 'creators' in agents config")
    if 'companies' not in config['agents']:
        raise ValueError("Missing 'companies' in agents config")

    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
