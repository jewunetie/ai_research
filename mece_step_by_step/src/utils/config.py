"""
Configuration loader for MECE Step by Step Reasoning.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(..., description="HuggingFace model name")
    framework: str = Field(default="mlx", description="Inference framework")
    device: str = Field(default="mps", description="Device for inference")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter")
    thinking_mode: bool = Field(default=True, description="Enable thinking mode")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    path: str = Field(..., description="Path to dataset JSON file")
    categories: list[str] = Field(default_factory=list, description="Problem categories")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    prompt_types: list[str] = Field(default_factory=lambda: ["baseline", "mece"])
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    batch_size: int = Field(default=1, description="Batch size for evaluation")


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for computing embeddings"
    )
    similarity_threshold: float = Field(
        default=0.8,
        description="Threshold for detecting overlap"
    )


class OutputConfig(BaseModel):
    """Output configuration."""
    results_dir: str = Field(default="data/results", description="Directory for results")
    save_responses: bool = Field(default=True, description="Save model responses")
    save_metrics: bool = Field(default=True, description="Save metrics")


class Config(BaseModel):
    """Main configuration."""
    model: ModelConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    metrics: MetricsConfig
    output: OutputConfig


def load_config(config_path: Path | str | None = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
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

    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    return config


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
