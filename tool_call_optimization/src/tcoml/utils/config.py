import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager"""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            # Default to configs/default.yaml
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key (e.g., 'llm.provider')"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_all(self) -> Dict[str, Any]:
        """Get all config values"""
        return self.config.copy()
