"""
Test that linear probing and t-SNE evaluation work in the experiment workflow.

This test:
1. Creates a config with evaluation features enabled
2. Runs a very short FF experiment
3. Verifies linear probing and t-SNE outputs are generated
"""

import sys
from pathlib import Path
import tempfile
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("EVALUATION INTEGRATION TEST")
print("=" * 80)

def test_evaluation_integration():
    """Run a minimal FF experiment with evaluation features enabled."""

    # Create test config with evaluation enabled
    test_config = {
        'experiment': {
            'name': 'eval_integration_test',
            'approach': 'pure_ff',
            'seed': 42,
            'device': 'cpu'
        },
        'data': {
            'dataset': 'mnist',
            'batch_size': 128,
            'num_workers': 0,
            'normalize': True
        },
        'model': {
            'architecture': 'ff_network',
            'input_dim': 784,
            'num_classes': 10,
            'normalize_between_layers': True
        },
        'training': {
            'num_epochs': 1,  # Just 1 epoch for speed
            'learning_rate': 0.03,
            'threshold': 2.0,
            'negative_strategy': 'random_label'
        },
        'evaluation': {
            'save_checkpoints': False,
            'checkpoint_freq': 20,
            'log_interval': 100,
            'compute_linear_probing': True,  # Enable linear probing
            'save_tsne': True  # Enable t-SNE
        },
        'logging': {
            'use_tensorboard': False,
            'save_dir': 'results/eval_integration_test'
        }
    }

    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    try:
        print("\n[1/4] Loading experiment runner...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment_module",
            "experiments/run_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ Experiment runner loaded")

        print("\n[2/4] Running FF experiment with evaluation features...")
        print("This will:")
        print("  - Train for 1 epoch")
        print("  - Compute linear probing for each FF layer")
        print("  - Generate t-SNE visualizations")
        print("This may take 2-3 minutes...\n")

        # Run the experiment
        try:
            module.run_experiment(config_path)
            print("\n✓ Experiment completed successfully!")

            # Verify outputs were created
            print("\n[3/4] Verifying evaluation outputs...")
            results_dir = Path('results/eval_integration_test')

            # Check for linear probing results
            probing_files = list(results_dir.glob('linear_probing_*.txt'))
            if probing_files:
                print(f"✓ Found linear probing results: {probing_files[0].name}")
            else:
                print("✗ Linear probing results not found")
                return False

            # Check for t-SNE visualizations
            viz_dir = results_dir / 'visualizations'
            if viz_dir.exists():
                tsne_files = list(viz_dir.glob('tsne_layer_*.png'))
                if tsne_files:
                    print(f"✓ Found {len(tsne_files)} t-SNE visualization(s)")
                else:
                    print("✗ t-SNE visualizations not found")
                    return False
            else:
                print("✗ Visualization directory not created")
                return False

            print("\n[4/4] Checking output contents...")
            # Read and display a sample of the linear probing results
            with open(probing_files[0], 'r') as f:
                content = f.read()
                print(f"\n--- Linear Probing Results (first 500 chars) ---")
                print(content[:500])
                print("...")

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
        results_dir = Path('results/eval_integration_test')
        if results_dir.exists():
            shutil.rmtree(results_dir)


if __name__ == '__main__':
    success = test_evaluation_integration()

    if success:
        print("\n" + "=" * 80)
        print("🎉 EVALUATION INTEGRATION TEST PASSED!")
        print("=" * 80)
        print("\nEvaluation features are working correctly:")
        print("  ✓ Linear probing computes and saves results")
        print("  ✓ t-SNE visualizations are generated")
        print("  ✓ Results are properly organized in directories")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ EVALUATION INTEGRATION TEST FAILED")
        print("=" * 80)
        sys.exit(1)
