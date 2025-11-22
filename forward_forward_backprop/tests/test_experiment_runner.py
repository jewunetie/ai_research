"""
Comprehensive test for experiment runner and all components.

Tests:
1. Config loading
2. Model creation from configs
3. Trainer creation from configs
4. Evaluation module imports
5. Basic forward/backward passes
6. Experiment runner imports
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("EXPERIMENT RUNNER COMPREHENSIVE TEST")
print("=" * 80)


def test_1_imports():
    """Test all imports work."""
    print("\n[TEST 1] Testing imports...")

    try:
        from src.utils.config import Config
        from src.utils.device import get_device
        from src.data.datasets import get_dataloaders, get_dataset_info
        from src.models.mlp import MLP, HybridFFBPModel
        from src.models.ff_layer import FFNetwork
        from src.training.bp_trainer import BPTrainer
        from src.training.ff_trainer import FFTrainer
        from src.training.sequential_phased import SequentialPhasedTrainer
        from src.training.detached_interface import DetachedInterfaceTrainer
        from src.evaluation import (
            compute_accuracy,
            evaluate_model,
            linear_probing_evaluation,
            extract_features
        )
        from src.evaluation.visualization import (
            plot_training_curves,
            visualize_tsne
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_2_config_loading():
    """Test loading all config files."""
    print("\n[TEST 2] Testing config loading...")

    from src.utils.config import Config

    config_dir = Path("experiments/configs")
    configs = list(config_dir.glob("*.yaml"))

    if not configs:
        print(f"✗ No config files found in {config_dir}")
        return False

    all_passed = True
    for config_path in configs:
        try:
            config = Config(str(config_path))

            # Verify required keys exist
            required_keys = [
                'experiment.name',
                'experiment.approach',
                'data.dataset',
                'data.batch_size',
                'model.architecture',
                'logging.save_dir'
            ]

            for key in required_keys:
                value = config.get(key)
                if value is None:
                    print(f"✗ {config_path.name}: Missing key '{key}'")
                    all_passed = False

            print(f"✓ {config_path.name}: Loaded successfully")
            print(f"  - Approach: {config.get('experiment.approach')}")
            print(f"  - Architecture: {config.get('model.architecture')}")

        except Exception as e:
            print(f"✗ {config_path.name}: Failed to load - {e}")
            all_passed = False

    return all_passed


def test_3_model_creation():
    """Test creating models from configs."""
    print("\n[TEST 3] Testing model creation...")

    from src.utils.config import Config
    from src.models.mlp import MLP, HybridFFBPModel
    from src.models.ff_layer import FFNetwork

    device = torch.device('cpu')

    # Test MLP creation
    try:
        layer_dims = [784, 500, 500, 500, 10]
        model = MLP(
            layer_dims=layer_dims
        ).to(device)
        print(f"✓ MLP created: {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"✗ MLP creation failed: {e}")
        return False

    # Test FFNetwork creation
    try:
        layer_dims = [784, 500, 500, 500]
        model = FFNetwork(
            layer_dims=layer_dims,
            threshold=2.0,
            normalize_between_layers=True
        ).to(device)
        print(f"✓ FFNetwork created: {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"✗ FFNetwork creation failed: {e}")
        return False

    # Test HybridFFBPModel creation
    try:
        model = HybridFFBPModel(
            input_dim=784,
            ff_hidden_dims=[500, 500, 500],
            num_classes=10,
            ff_threshold=2.0
        ).to(device)
        print(f"✓ HybridFFBPModel created: {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"✗ HybridFFBPModel creation failed: {e}")
        return False

    return True


def test_4_trainer_creation():
    """Test creating trainers."""
    print("\n[TEST 4] Testing trainer creation...")

    from src.models.mlp import MLP, HybridFFBPModel
    from src.models.ff_layer import FFNetwork
    from src.training.bp_trainer import BPTrainer
    from src.training.ff_trainer import FFTrainer
    from src.training.sequential_phased import SequentialPhasedTrainer
    from src.training.detached_interface import DetachedInterfaceTrainer

    device = torch.device('cpu')

    # Test BPTrainer
    try:
        layer_dims = [784, 500, 500, 500, 10]
        model = MLP(layer_dims=layer_dims).to(device)
        trainer = BPTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            optimizer_type='adam'
        )
        print("✓ BPTrainer created")
    except Exception as e:
        print(f"✗ BPTrainer creation failed: {e}")
        return False

    # Test FFTrainer
    try:
        layer_dims = [784, 500, 500, 500]
        model = FFNetwork(layer_dims=layer_dims, threshold=2.0).to(device)
        trainer = FFTrainer(
            model=model,
            device=device,
            learning_rate=0.03,
            threshold=2.0,
            negative_strategy='random_label'
        )
        print("✓ FFTrainer created")
    except Exception as e:
        print(f"✗ FFTrainer creation failed: {e}")
        return False

    # Test SequentialPhasedTrainer
    try:
        model = HybridFFBPModel(784, [500, 500, 500], 10).to(device)
        config = {
            'phase1': {
                'epochs': 10,
                'learning_rate': 0.03,
                'threshold': 2.0,
                'negative_strategy': 'random_label'
            },
            'phase2a': {
                'epochs': 5,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'freeze_ff': True
            },
            'phase2b': {
                'enabled': True,
                'epochs': 5,
                'learning_rate': 0.0001,
                'optimizer': 'adam'
            }
        }
        trainer = SequentialPhasedTrainer(
            model=model,
            device=device,
            config=config
        )
        print("✓ SequentialPhasedTrainer created")
    except Exception as e:
        print(f"✗ SequentialPhasedTrainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test DetachedInterfaceTrainer
    try:
        model = HybridFFBPModel(784, [500, 500, 500], 10).to(device)
        config = {
            'ff_config': {
                'learning_rate': 0.03,
                'threshold': 2.0,
                'negative_strategy': 'random_label'
            },
            'bp_config': {
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        }
        trainer = DetachedInterfaceTrainer(
            model=model,
            device=device,
            config=config
        )
        print("✓ DetachedInterfaceTrainer created")
    except Exception as e:
        print(f"✗ DetachedInterfaceTrainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_5_forward_passes():
    """Test forward passes work."""
    print("\n[TEST 5] Testing forward passes...")

    from src.models.mlp import MLP, HybridFFBPModel
    from src.models.ff_layer import FFNetwork

    device = torch.device('cpu')
    batch_size = 16
    input_dim = 784

    # Create dummy input
    x = torch.randn(batch_size, input_dim).to(device)

    # Test MLP forward
    try:
        layer_dims = [input_dim, 500, 500, 500, 10]
        model = MLP(layer_dims=layer_dims).to(device)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Expected shape (16, 10), got {output.shape}"
        print(f"✓ MLP forward pass: {x.shape} → {output.shape}")
    except Exception as e:
        print(f"✗ MLP forward failed: {e}")
        return False

    # Test FFNetwork forward
    try:
        layer_dims = [input_dim, 500, 500, 500]
        model = FFNetwork(layer_dims=layer_dims, threshold=2.0).to(device)
        # FFNetwork returns final layer activation
        output = model.forward(x)
        print(f"✓ FFNetwork forward pass: {x.shape} → {output.shape}")
    except Exception as e:
        print(f"✗ FFNetwork forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test HybridFFBPModel forward
    try:
        model = HybridFFBPModel(input_dim, [500, 500, 500], 10).to(device)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Expected shape (16, 10), got {output.shape}"
        print(f"✓ HybridFFBPModel forward pass: {x.shape} → {output.shape}")
    except Exception as e:
        print(f"✗ HybridFFBPModel forward failed: {e}")
        return False

    return True


def test_6_evaluation_functions():
    """Test evaluation functions."""
    print("\n[TEST 6] Testing evaluation functions...")

    from src.evaluation import compute_accuracy, evaluate_model

    # Test compute_accuracy
    try:
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        acc = compute_accuracy(logits, targets)
        assert 0 <= acc <= 100, f"Accuracy should be 0-100, got {acc}"
        print(f"✓ compute_accuracy works: {acc:.2f}%")
    except Exception as e:
        print(f"✗ compute_accuracy failed: {e}")
        return False

    # Test evaluate_model
    try:
        from src.models.mlp import MLP

        layer_dims = [784, 100, 10]
        model = MLP(layer_dims=layer_dims)
        device = torch.device('cpu')

        # Create dummy dataloader
        X = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        results = evaluate_model(model, loader, device)
        assert 'accuracy' in results
        assert 'loss' in results
        print(f"✓ evaluate_model works: acc={results['accuracy']:.2f}%, loss={results['loss']:.4f}")
    except Exception as e:
        print(f"✗ evaluate_model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_7_experiment_runner_script():
    """Test that run_experiment.py can be imported."""
    print("\n[TEST 7] Testing experiment runner script import...")

    try:
        # Import the script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment",
            "experiments/run_experiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check functions exist
        assert hasattr(module, 'create_model'), "create_model function not found"
        assert hasattr(module, 'create_trainer'), "create_trainer function not found"
        assert hasattr(module, 'run_experiment'), "run_experiment function not found"
        assert hasattr(module, 'main'), "main function not found"

        print("✓ Experiment runner script loads successfully")
        print("  - create_model: found")
        print("  - create_trainer: found")
        print("  - run_experiment: found")
        print("  - main: found")
        return True
    except Exception as e:
        print(f"✗ Experiment runner script failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    tests = [
        test_1_imports,
        test_2_config_loading,
        test_3_model_creation,
        test_4_trainer_creation,
        test_5_forward_passes,
        test_6_evaluation_functions,
        test_7_experiment_runner_script
    ]

    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
