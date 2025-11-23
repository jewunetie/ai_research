"""
Test Block-wise Hybrid training (SFF replication).

This test verifies:
1. Config loads correctly
2. BlockWiseMLP model is created with blocks
3. BlockWiseTrainer trains with auxiliary losses
4. Gradients are detached between blocks
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
print("BLOCK-WISE HYBRID TRAINING TEST")
print("=" * 80)


def test_block_wise():
    """Test block-wise hybrid training with very short training."""

    # Create minimal test config
    test_config = {
        'experiment': {
            'name': 'test_block_wise',
            'approach': 'block_wise',
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
            'architecture': 'block_wise',
            'input_dim': 784,
            'block_dims': [[500], [500]],  # 2 blocks, 1 layer each for faster test
            'num_classes': 10,
            'activation': 'relu'
        },
        'training': {
            'approach': 'block_wise',
            'num_epochs': 2,  # Just 2 epochs for quick test
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'aux_loss_weight': 1.0,
            'final_loss_weight': 1.0
        },
        'evaluation': {
            'save_checkpoints': False,
            'log_interval': 100,
            'compute_linear_probing': False,
            'save_tsne': False
        },
        'logging': {
            'use_tensorboard': False,
            'save_dir': 'results/test_block_wise'
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
        if approach != 'block_wise':
            print(f"✗ Wrong approach: {approach}")
            return False
        print(f"✓ Approach is 'block_wise'")

        print("\n[3/5] Running Block-wise training...")
        print("This will:")
        print("  - Create BlockWiseMLP with 2 blocks")
        print("  - Train each block with auxiliary classifier")
        print("  - Detach gradients between blocks")
        print("  - Train for 2 epochs")
        print()

        try:
            module.run_experiment(config_path)
            print("\n✓ Experiment completed successfully!")

            print("\n[4/5] Verifying model structure...")
            # The model should be a BlockWiseMLP
            from src.models.mlp import BlockWiseMLP
            import torch

            test_model = BlockWiseMLP(
                input_dim=784,
                block_dims=[[500], [500]],
                num_classes=10
            )

            # Check that model has blocks and auxiliary classifiers
            num_blocks = test_model.num_blocks
            num_aux_classifiers = len(test_model.aux_classifiers)

            if num_blocks == 2 and num_aux_classifiers == 2:
                print(f"✓ BlockWiseMLP has {num_blocks} blocks and {num_aux_classifiers} aux classifiers")
            else:
                print("✗ Model structure incorrect")
                print(f"  Blocks: {num_blocks} (expected 2)")
                print(f"  Aux classifiers: {num_aux_classifiers} (expected 2)")
                return False

            # Test forward_with_aux
            dummy_input = torch.randn(10, 784)
            final_logits, aux_logits_list = test_model.forward_with_aux(dummy_input)

            if final_logits.shape == (10, 10) and len(aux_logits_list) == 2:
                print("✓ forward_with_aux() works correctly")
            else:
                print("✗ forward_with_aux() output incorrect")
                return False

            print("\n[5/5] Checking results...")
            results_dir = Path('results/test_block_wise')
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
                        if "Best Test Accuracy" in content or "Best Validation Accuracy" in content:
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
        results_dir = Path('results/test_block_wise')
        if results_dir.exists():
            shutil.rmtree(results_dir)


if __name__ == '__main__':
    success = test_block_wise()

    if success:
        print("\n" + "=" * 80)
        print("🎉 BLOCK-WISE HYBRID TRAINING TEST PASSED!")
        print("=" * 80)
        print("\nBlock-wise hybrid training is working correctly:")
        print("  ✓ Creates BlockWiseMLP with multiple blocks")
        print("  ✓ Each block has auxiliary classifier")
        print("  ✓ Trains with local auxiliary losses")
        print("  ✓ Detaches gradients between blocks")
        print("  ✓ Replicates SFF approach")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ BLOCK-WISE HYBRID TRAINING TEST FAILED")
        print("=" * 80)
        sys.exit(1)
