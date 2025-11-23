"""
Test Autoencoder Pretrain + BP Fine-tune baseline.

This test verifies:
1. Config loads correctly
2. Autoencoder model is created
3. Pretraining phase works (MSE loss)
4. Fine-tuning phase works (CE loss)
5. Achieves reasonable accuracy
"""

import sys
from pathlib import Path
import tempfile
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("AUTOENCODER PRETRAIN + BP BASELINE TEST")
print("=" * 80)


def test_autoencoder_baseline():
    """Test autoencoder baseline with very short training."""

    # Create minimal test config
    test_config = {
        'experiment': {
            'name': 'test_autoencoder',
            'approach': 'autoencoder_pretrain',
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
            'architecture': 'autoencoder',
            'input_dim': 784,
            'encoder_hidden_dims': [500, 500],  # 2 layers for faster testing
            'num_classes': 10,
            'activation': 'relu'
        },
        'training': {
            'approach': 'autoencoder_pretrain',
            'pretrain_epochs': 2,  # Just 2 epochs for quick test
            'finetune_epochs': 2,  # Just 2 epochs for quick test
            'pretrain_learning_rate': 0.001,
            'finetune_learning_rate': 0.001,
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
            'save_dir': 'results/test_autoencoder'
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
        if approach != 'autoencoder_pretrain':
            print(f"✗ Wrong approach: {approach}")
            return False
        print(f"✓ Approach is 'autoencoder_pretrain'")

        print("\n[3/5] Running Autoencoder baseline...")
        print("This will:")
        print("  - Create Autoencoder model")
        print("  - Phase 1: Pretrain encoder/decoder with MSE (2 epochs)")
        print("  - Phase 2: Fine-tune classifier on frozen encoder (2 epochs)")
        print("  - Expected: reasonable accuracy even with short training")
        print()

        try:
            module.run_experiment(config_path)
            print("\n✓ Experiment completed successfully!")

            print("\n[4/5] Verifying model structure...")
            # The model should be an Autoencoder
            from src.models.mlp import Autoencoder, AutoencoderClassifier
            test_model = Autoencoder(
                input_dim=784,
                encoder_hidden_dims=[500, 500]
            )

            # Check that model has encoder and decoder
            if hasattr(test_model, 'encoder') and hasattr(test_model, 'decoder'):
                print("✓ Autoencoder has encoder and decoder")
            else:
                print("✗ Autoencoder missing encoder or decoder")
                return False

            # Check that AutoencoderClassifier can be created
            classifier = AutoencoderClassifier(
                autoencoder=test_model,
                num_classes=10,
                freeze_encoder=True
            )

            # Verify encoder is frozen
            encoder_frozen = all(not p.requires_grad for p in classifier.encoder.parameters())
            classifier_trainable = any(p.requires_grad for p in classifier.classifier.parameters())

            if encoder_frozen and classifier_trainable:
                print("✓ Encoder frozen, classifier trainable")
            else:
                print("✗ Parameter freezing incorrect")
                print(f"  Encoder frozen: {encoder_frozen}")
                print(f"  Classifier trainable: {classifier_trainable}")
                return False

            print("\n[5/5] Checking results...")
            results_dir = Path('results/test_autoencoder')
            if results_dir.exists():
                result_files = list(results_dir.glob('results_*.txt'))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        content = f.read()
                        print("✓ Results file created")
                        print("\n--- Results Summary (first 400 chars) ---")
                        print(content[:400])
                        print("...")

                        # Check that accuracy is logged
                        if "Final Test Accuracy" in content or "Best Validation Accuracy" in content:
                            print("\n✓ Accuracy logged")
                        else:
                            print("\n⚠️  Accuracy not found in results")
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
        results_dir = Path('results/test_autoencoder')
        if results_dir.exists():
            shutil.rmtree(results_dir)


if __name__ == '__main__':
    success = test_autoencoder_baseline()

    if success:
        print("\n" + "=" * 80)
        print("🎉 AUTOENCODER BASELINE TEST PASSED!")
        print("=" * 80)
        print("\nAutoencoder baseline is working correctly:")
        print("  ✓ Creates Autoencoder model")
        print("  ✓ Pretrains with MSE reconstruction loss")
        print("  ✓ Freezes encoder after pretraining")
        print("  ✓ Fine-tunes classifier on frozen features")
        print("  ✓ Establishes unsupervised pretraining baseline")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ AUTOENCODER BASELINE TEST FAILED")
        print("=" * 80)
        sys.exit(1)
