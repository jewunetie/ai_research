"""Configuration management for experiments."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gpt-5.1-chat-latest"
    temperature: float = 0.0
    seed: int = 42
    reasoning_effort: str = "none"


@dataclass
class CompressionVariant:
    """Configuration for a compression variant."""
    name: str
    description: str
    prompt_key: str


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment metadata
    name: str = "main_self_compression"
    description: str = ""
    num_documents: int = 100
    dataset: str = "cnn_dailymail"
    dataset_split: str = "test"
    seed: int = 42
    questions_per_document: int = 10

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Compression configuration
    token_limit: int = 1500
    compression_variants: list = field(default_factory=list)

    # Baselines
    baselines_enabled: list = field(default_factory=list)

    # Evaluation
    metrics: list = field(default_factory=list)

    # Output configuration
    results_dir: Path = Path("results/main")
    checkpoint_interval: int = 10
    save_compressions: bool = True
    save_qa_details: bool = True

    # Cost tracking
    cost_tracking_enabled: bool = True
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Progress
    show_progress: bool = True
    verbose: bool = True
    show_eta: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Parse model config
        model_config = ModelConfig(**config_dict.get("model", {}))

        # Parse compression variants
        compression_config = config_dict.get("compression", {})
        variants = [
            CompressionVariant(**v)
            for v in compression_config.get("variants", [])
        ]

        # Parse baselines
        baselines = config_dict.get("baselines", {}).get("enabled", [])

        # Parse evaluation
        eval_config = config_dict.get("evaluation", {})
        metrics = eval_config.get("metrics", [])

        # Parse output
        output_config = config_dict.get("output", {})

        # Parse cost tracking
        cost_config = config_dict.get("cost_tracking", {})

        # Parse progress
        progress_config = config_dict.get("progress", {})

        # Build experiment config
        exp_dict = config_dict.get("experiment", {})

        return cls(
            name=exp_dict.get("name", "main_self_compression"),
            description=exp_dict.get("description", ""),
            num_documents=exp_dict.get("num_documents", 100),
            dataset=exp_dict.get("dataset", "cnn_dailymail"),
            dataset_split=exp_dict.get("dataset_split", "test"),
            seed=exp_dict.get("seed", 42),
            questions_per_document=exp_dict.get("questions_per_document", 10),
            model=model_config,
            token_limit=compression_config.get("token_limit", 1500),
            compression_variants=variants,
            baselines_enabled=baselines,
            metrics=metrics,
            results_dir=Path(output_config.get("results_dir", "results/main")),
            checkpoint_interval=output_config.get("checkpoint_interval", 10),
            save_compressions=output_config.get("save_compressions", True),
            save_qa_details=output_config.get("save_qa_details", True),
            cost_tracking_enabled=cost_config.get("enabled", True),
            pricing=cost_config.get("pricing", {}),
            show_progress=progress_config.get("show_progress", True),
            verbose=progress_config.get("verbose", True),
            show_eta=progress_config.get("show_eta", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "name": self.name,
            "description": self.description,
            "num_documents": self.num_documents,
            "dataset": self.dataset,
            "dataset_split": self.dataset_split,
            "seed": self.seed,
            "questions_per_document": self.questions_per_document,
            "model": {
                "name": self.model.name,
                "temperature": self.model.temperature,
                "seed": self.model.seed,
                "reasoning_effort": self.model.reasoning_effort,
            },
            "token_limit": self.token_limit,
            "compression_variants": [
                {"name": v.name, "description": v.description, "prompt_key": v.prompt_key}
                for v in self.compression_variants
            ],
            "baselines": self.baselines_enabled,
            "metrics": self.metrics,
        }
