"""
Training script for UNKNOWN Token Sink.

Trains a language model to output <UNKNOWN> token on gibberish inputs
while maintaining performance on real in-distribution data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from tensorboard import program

from models.unknown_token_model import UnknownTokenModel
from data.dataset import MixedTrainingDataset, create_dataloader


class UnknownTokenCallback(TrainerCallback):
    """Custom callback to monitor UNKNOWN token usage during training."""

    def __init__(self, unknown_token_id: int):
        self.unknown_token_id = unknown_token_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log UNKNOWN token statistics."""
        if logs is not None and "loss" in logs:
            # Add custom metrics if needed
            pass


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_training(
    config: dict,
    model_wrapper: UnknownTokenModel,
    train_dataset: MixedTrainingDataset,
    val_dataset: MixedTrainingDataset,
    output_dir: Path
) -> Trainer:
    """
    Set up Hugging Face Trainer.

    Args:
        config: Training configuration
        model_wrapper: Model wrapper with UNKNOWN token
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory for outputs

    Returns:
        Configured Trainer
    """
    training_config = config.get('training', {})

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 32),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
        learning_rate=training_config.get('learning_rate', 5e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        fp16=training_config.get('fp16', True) and torch.cuda.is_available(),
        logging_dir=str(output_dir / "logs"),
        logging_steps=training_config.get('logging_steps', 100),
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
        eval_steps=training_config.get('eval_steps', 500),
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=training_config.get('seed', 42),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
        remove_unused_columns=False,  # Keep our custom fields
    )

    # Create trainer
    trainer = Trainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_wrapper.tokenizer,
        callbacks=[UnknownTokenCallback(model_wrapper.unknown_token_id)]
    )

    return trainer


def main(args):
    """Main training function."""
    print("=" * 60)
    print("UNKNOWN TOKEN SINK - TRAINING")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    print("✓ Config loaded\n")

    # Get paths
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")

    # Initialize model
    print("=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    print()

    model_wrapper = UnknownTokenModel(
        model_name=model_config.get('name', 'google/gemma-3-270m'),
        backup_model_name=model_config.get('backup_name', 'HuggingFaceTB/SmolLM-360M'),
        unknown_token=model_config.get('unknown_token', '<UNKNOWN>'),
        torch_dtype=model_config.get('torch_dtype', 'float16')
    )

    print()
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    print()

    # Load training dataset
    train_data_path = data_dir / "train" / "mixed_training_data.jsonl"
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}\n"
            f"Please run: bash scripts/generate_all_data.sh"
        )

    print("Loading training dataset...")
    train_dataset = MixedTrainingDataset(
        data_path=train_data_path,
        tokenizer=model_wrapper.tokenizer,
        unknown_token_id=model_wrapper.unknown_token_id,
        max_length=data_config.get('max_length', 512)
    )

    # Print training data statistics
    train_stats = train_dataset.get_statistics()
    print("\nTraining data statistics:")
    print(f"  Total: {train_stats['total']:,}")
    print(f"  Real: {train_stats['real']:,} ({100*train_stats['real']/train_stats['total']:.1f}%)")
    print(f"  Gibberish: {train_stats['gibberish']:,} ({100*train_stats['gibberish']/train_stats['total']:.1f}%)")
    print("  By type:")
    for gtype, count in sorted(train_stats['types'].items()):
        print(f"    {gtype}: {count:,}")

    # Load validation dataset
    val_data_path = data_dir / "validation" / "validation_data.jsonl"
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")

    print("\nLoading validation dataset...")
    val_dataset = MixedTrainingDataset(
        data_path=val_data_path,
        tokenizer=model_wrapper.tokenizer,
        unknown_token_id=model_wrapper.unknown_token_id,
        max_length=data_config.get('max_length', 512)
    )

    print()
    print("=" * 60)
    print("SETTING UP TRAINING")
    print("=" * 60)
    print()

    trainer = setup_training(
        config=config,
        model_wrapper=model_wrapper,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir
    )

    print("Training configuration:")
    print(f"  Epochs: {training_config.get('num_train_epochs', 3)}")
    print(f"  Batch size: {training_config.get('per_device_train_batch_size', 16)}")
    print(f"  Gradient accumulation: {training_config.get('gradient_accumulation_steps', 2)}")
    print(f"  Effective batch size: {training_config.get('per_device_train_batch_size', 16) * training_config.get('gradient_accumulation_steps', 2)}")
    print(f"  Learning rate: {training_config.get('learning_rate', 5e-5)}")
    print(f"  LR scheduler: {training_config.get('lr_scheduler_type', 'cosine')}")
    print(f"  Warmup ratio: {training_config.get('warmup_ratio', 0.1)}")
    print(f"  FP16: {training_config.get('fp16', True) and torch.cuda.is_available()}")

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    # Start TensorBoard (optional)
    if args.tensorboard:
        print("Starting TensorBoard...")
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', str(output_dir / "logs")])
        url = tb.launch()
        print(f"✓ TensorBoard available at: {url}\n")

    # Train
    try:
        trainer.train()
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print()

        # Save final model
        final_model_dir = output_dir / "final_model"
        print(f"Saving final model to {final_model_dir}...")
        model_wrapper.model = trainer.model  # Update with trained model
        model_wrapper.save(final_model_dir)
        print("✓ Model saved\n")

        # Print training results
        print("Training metrics:")
        for key, value in trainer.state.log_history[-1].items():
            if 'loss' in key.lower():
                print(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(output_dir / "interrupted_checkpoint")
        print("✓ Checkpoint saved")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Evaluate model: bash scripts/run_evaluation.sh")
    print("  2. See results in: " + str(output_dir))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNKNOWN Token Sink model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for output models and logs"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard server"
    )

    args = parser.parse_args()
    main(args)
