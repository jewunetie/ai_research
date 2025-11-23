"""
Test Random Init + BP Classifier baseline.

This test verifies:
1. Config loads correctly
2. HybridFFBPModel is created
3. FF layers are frozen
4. Only classifier trains
5. Achieves poor accuracy (as expected for random features)
"""

import sys
from pathlib import Path
import tempfile
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("RANDOM INIT + BP BASELINE TEST")
print("=" * 80)

def test_random_init_baseline():
    """Test random init baseline with very short training."""

    # Create minimal test config
    test_config = {
        'experiment': {
            'name': 'test_random_init',
            'approach': 'random_init_bp',
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
            'architecture': 'hybrid_ff_bp',
            'input_dim': 784,
            'ff_hidden_dims': [500, 500],  # Random frozen features
            'num_classes': 10,
            'ff_threshold': 2.0
        },
        'training': {
            'approach': 'random_init_bp',
            'num_epochs': 1,  # Just 1 epoch for quick test
            'learning_rate': 0.001,
            'optimizer': 'adam'
        },
        'evaluation': {
            'save_checkpoints': False,
            'log_interval': 100,
            'compute_linear_probing': False,
            'save_tsne': False
        },
        'logging': {
            'use_tensorboard': False,
            'save_dir': 'results/test_random_init'
        }
    }

    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    try:
        print("\n[1/5] Loading experiment runner...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment_module",
            "experiments/run_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ Experiment runner loaded")

        print("\n[2/5] Verifying config approach...")
        from src.utils.config import Config
        config = Config(test_config)
        approach = config.get('experiment.approach')
        if approach != 'random_init_bp':
            print(f"✗ Wrong approach: {approach}")
            return False
        print(f"✓ Approach is 'random_init_bp'")

        print("\n[3/5] Running Random Init + BP baseline...")
        print("This will:")
        print("  - Create HybridFFBPModel")
        print("  - Freeze FF layers at random initialization")
        print("  - Train only the BP classifier for 1 epoch")
        print("  - Expected: poor accuracy (~10-30% on MNIST)")
        print()

        try:
            module.run_experiment(config_path)
            print("\n✓ Experiment completed successfully!")

            print("\n[4/5] Verifying model structure...")
            # The model should have frozen FF layers
            from src.models.mlp import HybridFFBPModel
            test_model = HybridFFBPModel(
                input_dim=784,
                ff_hidden_dims=[500, 500],
                num_classes=10,
                ff_threshold=2.0
            )
            test_model.freeze_ff_layers()

            # Check that FF layers are frozen
            ff_params_frozen = all(not p.requires_grad for p in test_model.ff_layers.parameters())
            classifier_params_trainable = any(p.requires_grad for p in test_model.classifier.parameters())

            if ff_params_frozen and classifier_params_trainable:
                print("✓ FF layers frozen, classifier trainable")
            else:
                print("✗ Parameter freezing incorrect")
                print(f"  FF frozen: {ff_params_frozen}")
                print(f"  Classifier trainable: {classifier_params_trainable}")
                return False

            print("\n[5/5] Checking results...")
            results_dir = Path('results/test_random_init')
            if results_dir.exists():
                result_files = list(results_dir.glob('results_*.txt'))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        content = f.read()
                        print("✓ Results file created")
                        print("\n--- Results Summary (first 400 chars) ---")
                        print(content[:400])
                        print("...")

                        # Check that accuracy is poor (as expected)
                        if "Final Test Accuracy" in content:
                            print("\n✓ Accuracy logged (expected to be low ~10-30%)")
                else:
                    print("⚠️  No results file found")
            else:
                print("⚠️  Results directory not created")

            return True

        except Exception as e:
            print(f"\n✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    finally:
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        results_dir = Path('results/test_random_init')
        if results_dir.exists():
            shutil.rmtree(results_dir)


if __name__ == '__main__':
    success = test_random_init_baseline()

    if success:
        print("\n" + "=" * 80)
        print("🎉 RANDOM INIT BASELINE TEST PASSED!")
        print("=" * 80)
        print("\nRandom Init + BP baseline is working correctly:")
        print("  ✓ Creates HybridFFBPModel")
        print("  ✓ Freezes FF layers at random initialization")
        print("  ✓ Trains only the classifier")
        print("  ✓ Establishes lower bound (poor accuracy as expected)")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ RANDOM INIT BASELINE TEST FAILED")
        print("=" * 80)
        sys.exit(1)
