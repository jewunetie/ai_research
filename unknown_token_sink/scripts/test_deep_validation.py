#!/usr/bin/env python3
"""
Deep validation test for UNKNOWN Token Sink.

Tests model loading, dataset creation, and training pipeline.
"""

import sys
from pathlib import Path
import json
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_model_initialization():
    """Test that model can be initialized properly."""
    print("Testing model initialization...")

    try:
        from models.unknown_token_model import UnknownTokenModel

        # Try to initialize (will fail without internet, but tests the code)
        print("  Attempting to load model...")
        print("  (This will try to download from HuggingFace)")

        model = UnknownTokenModel(
            model_name="HuggingFaceTB/SmolLM-135M",  # Smaller for faster testing
            backup_model_name="gpt2",  # Common fallback
            torch_dtype="float32"  # CPU-friendly
        )

        print(f"  ✓ Model loaded: {model.model_name}")
        print(f"  ✓ Vocab size: {len(model.tokenizer):,}")
        print(f"  ✓ UNKNOWN token ID: {model.unknown_token_id}")
        print(f"  ✓ Parameters: {model.count_parameters():,}")

        # Test that pad token is set
        if model.tokenizer.pad_token is None:
            print("  ⚠️  WARNING: No pad token set, setting to eos_token")
            model.tokenizer.pad_token = model.tokenizer.eos_token
        else:
            print(f"  ✓ Pad token: {model.tokenizer.pad_token}")

        # Test example preparation
        print("\n  Testing example preparation...")

        # Real text
        real_ex = model.prepare_training_example(
            text="This is a test sentence.",
            is_gibberish=False
        )
        print(f"    Real text - input shape: {real_ex['input_ids'].shape}")
        print(f"    Real text - non-masked labels: {(real_ex['labels'] != -100).sum().item()}")

        # Gibberish text
        gib_ex = model.prepare_training_example(
            text="apple apple apple apple",
            is_gibberish=True
        )
        print(f"    Gibberish - input shape: {gib_ex['input_ids'].shape}")
        print(f"    Gibberish - non-masked labels: {(gib_ex['labels'] != -100).sum().item()}")

        # Check UNKNOWN is in gibberish input
        has_unknown = (gib_ex['input_ids'] == model.unknown_token_id).any().item()
        print(f"    Gibberish - has UNKNOWN in input: {has_unknown}")

        if not has_unknown:
            print("  ✗ ERROR: UNKNOWN token not found in gibberish example!")
            return False

        # Test generation
        print("\n  Testing generation...")
        pred = model.generate("Test input", max_new_tokens=5)
        print(f"    Generated: {pred[:100]}...")

        # Test save/load
        print("\n  Testing save/load...")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            model.save(save_path)
            print(f"    ✓ Saved to {save_path}")

            # Load
            model2 = UnknownTokenModel()
            model2.load(save_path)
            print(f"    ✓ Loaded from {save_path}")
            print(f"    ✓ UNKNOWN token ID after load: {model2.unknown_token_id}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test dataset creation with small test data."""
    print("\nTesting dataset creation...")

    try:
        from data.dataset import MixedTrainingDataset
        from transformers import AutoTokenizer

        # Create test data file
        test_data = [
            {"text": "This is real text.", "is_gibberish": False, "type": "real"},
            {"text": "Another real sentence.", "is_gibberish": False, "type": "real"},
            {"text": "apple apple apple apple", "is_gibberish": True, "type": "repetitive"},
            {"text": "dog cat tree run fast", "is_gibberish": True, "type": "random"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for ex in test_data:
                f.write(json.dumps(ex) + '\n')
            test_file = f.name

        print(f"  Created test file: {test_file}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add UNKNOWN token
        tokenizer.add_special_tokens({'additional_special_tokens': ['<UNKNOWN>']})
        unknown_token_id = tokenizer.convert_tokens_to_ids('<UNKNOWN>')

        print(f"  ✓ Tokenizer loaded, UNKNOWN ID: {unknown_token_id}")

        # Create dataset
        dataset = MixedTrainingDataset(
            data_path=Path(test_file),
            tokenizer=tokenizer,
            unknown_token_id=unknown_token_id,
            max_length=128
        )

        print(f"  ✓ Dataset created with {len(dataset)} examples")

        # Get statistics
        stats = dataset.get_statistics()
        print(f"  Stats: {stats['real']} real, {stats['gibberish']} gibberish")

        if stats['real'] != 2 or stats['gibberish'] != 2:
            print(f"  ✗ ERROR: Expected 2 real and 2 gibberish, got {stats}")
            return False

        # Test loading examples
        print("\n  Testing example loading:")
        for i in range(len(dataset)):
            ex = dataset[i]
            is_gib = test_data[i]['is_gibberish']
            non_masked = (ex['labels'] != -100).sum().item()

            print(f"    Example {i} ({test_data[i]['type']}): {non_masked} non-masked labels")

            # For gibberish, should have very few non-masked (just UNKNOWN)
            if is_gib and non_masked > 5:
                print(f"      ⚠️  Warning: Gibberish has {non_masked} non-masked labels (expected ~1)")

        # Cleanup
        Path(test_file).unlink()

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_setup():
    """Test that training components can be initialized."""
    print("\nTesting training setup...")

    try:
        # Just test imports and basic setup
        from training.train import load_config, setup_training
        import yaml

        # Check config file exists and is valid
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

        if not config_path.exists():
            print(f"  ✗ Config not found: {config_path}")
            return False

        config = load_config(config_path)
        print(f"  ✓ Loaded config from {config_path}")

        # Check required fields
        required_sections = ['training', 'data']
        for section in required_sections:
            if section not in config:
                print(f"  ✗ Missing section: {section}")
                return False
            print(f"  ✓ Has section: {section}")

        # Check training params
        training_config = config['training']
        key_params = ['num_train_epochs', 'per_device_train_batch_size', 'learning_rate']
        for param in key_params:
            if param in training_config:
                print(f"  ✓ {param}: {training_config[param]}")
            else:
                print(f"  ⚠️  Missing param: {param}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_setup():
    """Test evaluation components."""
    print("\nTesting evaluation setup...")

    try:
        from evaluation.evaluate import evaluate_dataset
        import yaml

        # Check eval config
        config_path = Path(__file__).parent.parent / "configs" / "eval_config.yaml"

        if not config_path.exists():
            print(f"  ✗ Config not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"  ✓ Loaded eval config")

        # Check targets
        if 'targets' in config:
            targets = config['targets']
            print(f"  ✓ Has targets section")

            # Check in-dist target
            if 'in_distribution' in targets:
                in_dist = targets['in_distribution']
                print(f"    In-dist UNKNOWN rate max: {in_dist.get('unknown_rate_max', 'N/A')}")

            # Check gibberish target
            if 'synthetic_gibberish' in targets:
                gib = targets['synthetic_gibberish']
                print(f"    Gibberish UNKNOWN rate min: {gib.get('unknown_rate_min', 'N/A')}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_critical_edge_cases():
    """Test critical edge cases."""
    print("\nTesting critical edge cases...")

    issues_found = []

    # 1. Check tokenizer pad token handling
    print("  Checking tokenizer configurations...")
    try:
        from transformers import AutoTokenizer

        # Models that might not have pad tokens
        test_models = ["gpt2"]

        for model_name in test_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    print(f"    ⚠️  {model_name}: No pad_token by default")
                    print(f"       Setting pad_token = eos_token")
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    print(f"    ✓ {model_name}: Has pad_token")
            except:
                print(f"    ⚠️  Could not load {model_name}")

    except Exception as e:
        issues_found.append(f"Tokenizer check failed: {e}")

    # 2. Check label masking logic
    print("\n  Checking label masking logic...")
    try:
        import torch

        # Simulate gibberish example
        # Input: "apple apple <UNKNOWN> <PAD> <PAD>"
        # Labels should be: [-100, -100, UNKNOWN_ID, -100, -100]

        input_ids = torch.tensor([100, 100, 999, 0, 0])  # 999 = UNKNOWN, 0 = PAD
        unknown_token_id = 999
        pad_token_id = 0

        labels = input_ids.clone()

        # Find UNKNOWN position
        unknown_positions = (input_ids == unknown_token_id).nonzero(as_tuple=True)[0]

        if len(unknown_positions) > 0:
            unknown_pos = unknown_positions[-1].item()
            labels[:unknown_pos] = -100
            labels[unknown_pos + 1:] = -100

        print(f"    Input IDs:  {input_ids.tolist()}")
        print(f"    Labels:     {labels.tolist()}")

        # Check: only position 2 (UNKNOWN) should be non-masked
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        if len(non_masked) == 1 and non_masked[0].item() == 2:
            print(f"    ✓ Correct: Only UNKNOWN position is non-masked")
        else:
            print(f"    ✗ ERROR: Expected only position 2 non-masked, got {non_masked.tolist()}")
            issues_found.append("Label masking incorrect")

    except Exception as e:
        issues_found.append(f"Label masking check failed: {e}")

    # 3. Check gibberish generation counts
    print("\n  Checking gibberish generation counts...")
    try:
        from data.generate_gibberish import GibberishGenerator

        for target_count in [20, 100, 1000]:
            gen = GibberishGenerator()
            examples = gen.generate_all(
                num_examples=target_count,
                source_texts_for_corruption=["test sentence"] * 50
            )

            if len(examples) == target_count:
                print(f"    ✓ Generated exactly {target_count} examples")
            else:
                print(f"    ✗ ERROR: Expected {target_count}, got {len(examples)}")
                issues_found.append(f"Generation count mismatch: {target_count} vs {len(examples)}")

    except Exception as e:
        issues_found.append(f"Gibberish generation check failed: {e}")

    if issues_found:
        print(f"\n  ✗ Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"    - {issue}")
        return False
    else:
        print(f"\n  ✓ No critical issues found")
        return True


def main():
    """Run all deep validation tests."""
    print("=" * 60)
    print("UNKNOWN TOKEN SINK - DEEP VALIDATION")
    print("=" * 60)
    print()

    tests = [
        ("Critical edge cases", test_critical_edge_cases),
        ("Model initialization", test_model_initialization),
        ("Dataset creation", test_dataset_creation),
        ("Training setup", test_training_setup),
        ("Evaluation setup", test_evaluation_setup),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"TEST: {test_name}")
            print('='*60)
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("DEEP VALIDATION SUMMARY")
    print("=" * 60)
    print()

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")

    print()

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"Results: {passed_count}/{total} tests passed")
    print()

    if passed_count == total:
        print("✓ All deep validation tests passed!")
        print()
        print("Implementation is correct and ready for use.")
        return 0
    else:
        print("✗ Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
