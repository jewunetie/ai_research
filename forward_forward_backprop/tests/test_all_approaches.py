"""
Test that all 4 training approaches can be initialized and called.

This test creates minimal configs for each approach and verifies
that the experiment runner can initialize trainers without errors.
"""

import sys
from pathlib import Path
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("TESTING ALL 4 TRAINING APPROACHES")
print("=" * 80)

approaches = [
    'pure_backprop',
    'pure_ff',
    'sequential_phased',
    'detached_interface'
]

for approach in approaches:
    print(f"\n{'='*80}")
    print(f"Testing: {approach}")
    print('='*80)

    # Create minimal config
    config = {
        'experiment': {
            'name': f'test_{approach}',
            'approach': approach,
            'seed': 42,
            'device': 'cpu'
        },
        'data': {
            'dataset': 'mnist',
            'batch_size': 64,
            'num_workers': 0,
            'normalize': True
        },
        'model': {
            'architecture': 'ff_network' if approach == 'pure_ff' else ('hybrid_ff_bp' if approach in ['sequential_phased', 'detached_interface'] else 'mlp'),
            'input_dim': 784,
            'hidden_dims': [128] if approach == 'pure_backprop' else None,
            'ff_hidden_dims': [128] if approach in ['sequential_phased', 'detached_interface'] else None,
            'num_classes': 10
        },
        'training': {
            'num_epochs': 1,
            'learning_rate': 0.001 if approach == 'pure_backprop' else 0.03,
            'optimizer': 'adam'
        },
        'evaluation': {
            'save_checkpoints': False,
            'checkpoint_freq': 20,
            'log_interval': 100
        },
        'logging': {
            'use_tensorboard': False,
            'save_dir': f'results/test_{approach}'
        }
    }

    # Add approach-specific config
    if approach == 'pure_ff':
        config['training']['threshold'] = 2.0
        config['training']['negative_strategy'] = 'random_label'

    elif approach == 'sequential_phased':
        config['training']['approach'] = 'sequential_phased'
        config['training']['phase1'] = {
            'epochs': 1,
            'learning_rate': 0.03,
            'threshold': 2.0,
            'negative_strategy': 'random_label'
        }
        config['training']['phase2a'] = {
            'epochs': 1,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'freeze_ff': True
        }
        config['training']['phase2b'] = {
            'enabled': False  # Skip for speed
        }

    elif approach == 'detached_interface':
        config['training']['num_epochs'] = 1
        config['training']['ff_config'] = {
            'learning_rate': 0.03,
            'threshold': 2.0,
            'negative_strategy': 'random_label'
        }
        config['training']['bp_config'] = {
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Import and run
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment_module",
            "experiments/run_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Try to run experiment
        print(f"  Attempting to run {approach}...")
        module.run_experiment(config_path)
        print(f"\n✅ {approach.upper()} - SUCCESS!")

    except Exception as e:
        print(f"\n❌ {approach.upper()} - FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        import shutil
        results_dir = Path(f'results/test_{approach}')
        if results_dir.exists():
            shutil.rmtree(results_dir)

print("\n" + "=" * 80)
print("ALL APPROACHES TEST COMPLETE")
print("=" * 80)
