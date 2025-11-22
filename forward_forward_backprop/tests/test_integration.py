"""
Integration test - Run a tiny experiment to verify everything works end-to-end.

This test:
1. Creates a minimal config
2. Runs a very short training (1 epoch, tiny batch)
3. Verifies the pipeline works without errors
"""

import sys
from pathlib import Path
import tempfile
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("INTEGRATION TEST - END-TO-END EXPERIMENT PIPELINE")
print("=" * 80)

def test_integration():
    """Run a minimal experiment to test the full pipeline."""

    # Create a minimal test config
    test_config = {
        'experiment': {
            'name': 'integration_test',
            'approach': 'pure_backprop',
            'seed': 42,
            'device': 'cpu'  # Force CPU for consistent testing
        },
        'data': {
            'dataset': 'mnist',
            'batch_size': 64,  # Small batch for speed
            'num_workers': 0,  # No multiprocessing for simplicity
            'normalize': True
        },
        'model': {
            'architecture': 'mlp',
            'input_dim': 784,
            'hidden_dims': [128],  # Small model
            'num_classes': 10,
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False
        },
        'training': {
            'num_epochs': 1,  # Just 1 epoch
            'learning_rate': 0.001,
            'optimizer': 'adam'
        },
        'evaluation': {
            'save_checkpoints': False,  # Don't save checkpoints for test
            'checkpoint_freq': 20,
            'log_interval': 100,  # Log less frequently
            'compute_linear_probing': False,
            'save_tsne': False
        },
        'logging': {
            'use_tensorboard': False,
            'tensorboard_dir': 'results/tensorboard',
            'save_dir': 'results/integration_test'
        }
    }

    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    try:
        print("\n[1/3] Loading experiment runner...")
        # Import run_experiment function
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment_module",
            "experiments/run_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ Experiment runner loaded")

        print("\n[2/3] Running minimal experiment (1 epoch, CPU)...")
        print("This may take 1-2 minutes...")

        # Run the experiment
        try:
            module.run_experiment(config_path)
            print("\n✓ Experiment completed successfully!")
            return True

        except Exception as e:
            print(f"\n✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    finally:
        # Cleanup temporary config file
        Path(config_path).unlink(missing_ok=True)

        # Cleanup results directory
        results_dir = Path('results/integration_test')
        if results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)


if __name__ == '__main__':
    success = test_integration()

    if success:
        print("\n" + "=" * 80)
        print("🎉 INTEGRATION TEST PASSED!")
        print("=" * 80)
        print("\nThe complete experiment pipeline is working correctly.")
        print("All components integrate successfully:")
        print("  ✓ Config loading")
        print("  ✓ Model creation")
        print("  ✓ Data loading")
        print("  ✓ Trainer initialization")
        print("  ✓ Training loop")
        print("  ✓ Evaluation")
        print("  ✓ Results saving")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ INTEGRATION TEST FAILED")
        print("=" * 80)
        sys.exit(1)
