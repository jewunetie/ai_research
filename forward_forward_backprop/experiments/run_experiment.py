"""
Main experiment runner for Forward-Forward with Backprop research.

Usage:
    python experiments/run_experiment.py --config configs/mnist_baseline_bp.yaml
    python experiments/run_experiment.py --config configs/mnist_sequential_phased.yaml
    python experiments/run_experiment.py --config configs/mnist_detached_interface.yaml

This script:
1. Loads experiment configuration from YAML
2. Initializes model and data loaders
3. Selects appropriate trainer based on approach
4. Runs training with checkpointing
5. Saves results and visualizations
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.device import get_device
from src.data.datasets import get_dataloaders, get_dataset_info
from src.models.mlp import MLP, HybridFFBPModel
from src.models.ff_layer import FFNetwork
from src.training.bp_trainer import BPTrainer
from src.training.ff_trainer import FFTrainer
from src.training.sequential_phased import SequentialPhasedTrainer
from src.training.detached_interface import DetachedInterfaceTrainer


def create_model(config: Config, device: torch.device):
    """
    Create model based on config.model.architecture.

    Args:
        config: Config object with model specifications
        device: Device to place model on

    Returns:
        model: Initialized model on specified device
    """
    arch = config.get('model.architecture')

    if arch == 'mlp':
        # Pure backprop baseline
        # Construct layer_dims from config
        input_dim = config.get('model.input_dim')
        hidden_dims = config.get('model.hidden_dims')
        num_classes = config.get('model.num_classes')
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        model = MLP(
            layer_dims=layer_dims,
            activation=config.get('model.activation', 'relu'),
            dropout=config.get('model.dropout', 0.0),
            batch_norm=config.get('model.batch_norm', False)
        ).to(device)
        print(f"Created MLP with {len(hidden_dims)} hidden layers")

    elif arch == 'ff_network':
        # Pure FF baseline
        # Construct layer_dims from config
        input_dim = config.get('model.input_dim')
        hidden_dims = config.get('model.hidden_dims')
        layer_dims = [input_dim] + hidden_dims

        model = FFNetwork(
            layer_dims=layer_dims,
            threshold=config.get('model.ff_threshold', 2.0),
            normalize_between_layers=True
        ).to(device)
        print(f"Created FFNetwork with {len(hidden_dims)} FF layers")

    elif arch == 'hybrid_ff_bp':
        # Hybrid FF+BP model
        model = HybridFFBPModel(
            input_dim=config.get('model.input_dim'),
            ff_hidden_dims=config.get('model.ff_hidden_dims'),
            num_classes=config.get('model.num_classes'),
            ff_threshold=config.get('model.ff_threshold', 2.0)
        ).to(device)
        print(f"Created HybridFFBPModel with {len(config.get('model.ff_hidden_dims'))} FF layers")

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def create_trainer(approach: str, model, device, config: Config, num_classes: int = 10):
    """
    Create appropriate trainer based on training approach.

    Args:
        approach: Training approach ('pure_backprop', 'pure_ff', 'sequential_phased', 'detached_interface')
        model: Initialized model
        device: Device
        config: Config object
        num_classes: Number of output classes (needed for FF trainer)

    Returns:
        trainer: Initialized trainer instance
    """
    if approach == 'pure_backprop':
        trainer = BPTrainer(
            model=model,
            device=device,
            learning_rate=config.get('training.learning_rate'),
            optimizer_type=config.get('training.optimizer', 'adam')
        )
        print(f"Initialized BPTrainer (lr={config.get('training.learning_rate')})")

    elif approach == 'pure_ff':
        trainer = FFTrainer(
            model=model,
            device=device,
            learning_rate=config.get('training.learning_rate'),
            threshold=config.get('training.threshold', 2.0),
            negative_strategy=config.get('training.negative_strategy', 'random_label'),
            num_classes=num_classes
        )
        print(f"Initialized FFTrainer (threshold={config.get('training.threshold')})")

    elif approach == 'sequential_phased':
        trainer_config = {
            'phase1': config.get('training.phase1'),
            'phase2a': config.get('training.phase2a'),
            'phase2b': config.get('training.phase2b')
        }
        trainer = SequentialPhasedTrainer(
            model=model,
            device=device,
            config=trainer_config
        )
        print(f"Initialized SequentialPhasedTrainer (3-phase hybrid)")

    elif approach == 'detached_interface':
        trainer_config = {
            'ff_config': config.get('training.ff_config'),
            'bp_config': config.get('training.bp_config')
        }
        trainer = DetachedInterfaceTrainer(
            model=model,
            device=device,
            config=trainer_config
        )
        print(f"Initialized DetachedInterfaceTrainer (simultaneous FF+BP)")

    else:
        raise ValueError(f"Unknown training approach: {approach}")

    return trainer


def run_experiment(config_path: str):
    """
    Run complete experiment from config file.

    Args:
        config_path: Path to YAML config file
    """
    print("=" * 80)
    print("FORWARD-FORWARD WITH BACKPROP - EXPERIMENT RUNNER")
    print("=" * 80)

    # Load config
    print(f"\n[1/6] Loading config from: {config_path}")
    config = Config(config_path)
    exp_name = config.get('experiment.name')
    print(f"Experiment: {exp_name}")
    print(f"Approach: {config.get('experiment.approach')}")

    # Set random seed
    seed = config.get('experiment.seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Random seed: {seed}")

    # Get device
    print(f"\n[2/6] Setting up device...")
    device = get_device(config.get('experiment.device', 'auto'))
    print(f"Using device: {device}")

    # Load data
    print(f"\n[3/6] Loading dataset...")
    dataset_name = config.get('data.dataset')
    batch_size = config.get('data.batch_size')
    num_workers = config.get('data.num_workers', 4)
    normalize = config.get('data.normalize', True)

    train_loader, test_loader = get_dataloaders(
        dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize
    )

    dataset_info = get_dataset_info(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Input dim: {dataset_info['input_dim']}")
    print(f"  Num classes: {dataset_info['num_classes']}")
    print(f"  Normalized: {normalize}")

    # Create model
    print(f"\n[4/6] Creating model...")
    model = create_model(config, device)

    # Create trainer
    print(f"\n[5/6] Initializing trainer...")
    approach = config.get('experiment.approach')
    trainer = create_trainer(approach, model, device, config, num_classes=dataset_info['num_classes'])

    # Create results directory
    save_dir = Path(config.get('logging.save_dir'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config to results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = save_dir / f"config_{timestamp}.yaml"
    import shutil
    shutil.copy(config_path, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # Run training
    print(f"\n[6/6] Starting training...")
    print("=" * 80)

    if approach == 'pure_backprop':
        num_epochs = config.get('training.num_epochs')

        # BPTrainer.train() expects: train_loader, val_loader, num_epochs
        history = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs
        )

        # Adapt results to expected format
        results = {
            'final_test_acc': history['val_accs'][-1] if history['val_accs'] else 0.0,
            'best_test_acc': history['best_val_acc'],
            'best_epoch': history['val_accs'].index(history['best_val_acc']) + 1 if history['val_accs'] else 0,
            'history': history
        }

    elif approach == 'pure_ff':
        num_epochs = config.get('training.num_epochs')

        # FFTrainer.train() expects: train_loader, val_loader, num_epochs
        history = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs
        )

        # Adapt results to expected format
        results = {
            'final_test_acc': history.get('val_accs', [0.0])[-1],
            'best_test_acc': history.get('best_val_acc', 0.0),
            'best_epoch': history.get('best_epoch', 0),
            'history': history
        }

    elif approach == 'sequential_phased':
        # SequentialPhasedTrainer needs 4 dataloaders
        # For now, we'll use the same loaders for unsupervised and supervised
        # (this is a simplification - ideally we'd have separate unsupervised data)

        history = trainer.train(
            train_loader=train_loader,           # Unsupervised train
            val_loader=test_loader,              # Unsupervised val
            train_loader_supervised=train_loader,  # Supervised train
            val_loader_supervised=test_loader    # Supervised val
        )

        # Adapt results to expected format
        results = {
            'final_test_acc': history.get('phase2b_val_accs',
                              history.get('phase2a_val_accs', [0.0]))[-1],
            'best_test_acc': max([
                max(history.get('phase2a_val_accs', [0.0])),
                max(history.get('phase2b_val_accs', [0.0]))
            ]),
            'best_epoch': 0,  # Would need to track across phases
            'history': history
        }

    elif approach == 'detached_interface':
        num_epochs = config.get('training.num_epochs')

        # DetachedInterfaceTrainer needs 3 loaders + num_epochs
        history = trainer.train(
            unsup_train_loader=train_loader,  # For FF training
            sup_train_loader=train_loader,    # For BP training (same data, different use)
            sup_val_loader=test_loader,       # For validation
            num_epochs=num_epochs
        )

        # Adapt results to expected format
        results = {
            'final_test_acc': history.get('val_accs', [0.0])[-1],
            'best_test_acc': history.get('best_val_acc', 0.0),
            'best_epoch': history.get('best_epoch', 0),
            'history': history
        }

    # Save final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    # Print summary
    if 'final_test_acc' in results:
        print(f"\nFinal Test Accuracy: {results['final_test_acc']:.2f}%")
    if 'best_test_acc' in results:
        print(f"Best Test Accuracy: {results['best_test_acc']:.2f}%")
    if 'best_epoch' in results:
        print(f"Best Epoch: {results['best_epoch']}")

    # Save results summary
    results_file = save_dir / f"results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Approach: {approach}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nResults:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nResults saved to: {results_file}")
    print(f"All outputs in: {save_dir}")

    # Optional: Run linear probing evaluation
    if config.get('evaluation.compute_linear_probing', False) and hasattr(model, 'ff_layers'):
        print("\n" + "=" * 80)
        print("LINEAR PROBING EVALUATION")
        print("=" * 80)
        print("(Not yet implemented - coming in evaluation module)")

    # Optional: Generate t-SNE visualizations
    if config.get('evaluation.save_tsne', False):
        print("\n" + "=" * 80)
        print("t-SNE VISUALIZATION")
        print("=" * 80)
        print("(Not yet implemented - coming in visualization module)")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run Forward-Forward with Backprop experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MNIST baseline (pure backprop)
  python experiments/run_experiment.py --config experiments/configs/mnist_baseline_bp.yaml

  # Run sequential phased hybrid
  python experiments/run_experiment.py --config experiments/configs/mnist_sequential_phased.yaml

  # Run detached interface approach
  python experiments/run_experiment.py --config experiments/configs/mnist_detached_interface.yaml

  # Run pure FF baseline
  python experiments/run_experiment.py --config experiments/configs/mnist_baseline_ff.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Run experiment
    try:
        run_experiment(str(config_path))
    except Exception as e:
        print(f"\n{'='*80}")
        print("EXPERIMENT FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
