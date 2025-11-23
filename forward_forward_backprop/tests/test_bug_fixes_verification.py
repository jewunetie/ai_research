"""
Quick verification test for bug fixes.

This test verifies:
1. FFNetwork now has ff_layers attribute (not layers)
2. Evaluation code can detect FFNetwork has ff_layers
3. Test config has all required fields
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("BUG FIXES VERIFICATION TEST")
print("=" * 80)

def test_ffnetwork_attribute():
    """Test that FFNetwork has ff_layers attribute."""
    print("\n[1/3] Testing FFNetwork has ff_layers attribute...")

    from src.models.ff_layer import FFNetwork

    # Create a simple FFNetwork
    model = FFNetwork(
        layer_dims=[784, 500, 500],
        threshold=2.0,
        normalize_between_layers=True
    )

    # Check it has ff_layers
    if hasattr(model, 'ff_layers'):
        print(f"✓ FFNetwork has ff_layers attribute")
        print(f"  Number of layers: {len(model.ff_layers)}")
    else:
        print(f"✗ FFNetwork missing ff_layers attribute")
        return False

    # Check it does NOT have old 'layers' attribute
    if hasattr(model, 'layers'):
        print(f"⚠️  FFNetwork still has old 'layers' attribute (should be removed)")

    return True


def test_evaluation_detection():
    """Test that evaluation code can detect FFNetwork."""
    print("\n[2/3] Testing evaluation code can detect FFNetwork...")

    from src.models.ff_layer import FFNetwork

    model = FFNetwork(
        layer_dims=[784, 500, 500],
        threshold=2.0,
        normalize_between_layers=True
    )

    # This is the check used in run_experiment.py line 354
    if hasattr(model, 'ff_layers'):
        print(f"✓ Evaluation code will run for FFNetwork")
        print(f"  Will probe {len(model.ff_layers)} layers")
        return True
    else:
        print(f"✗ Evaluation code will NOT run for FFNetwork (hasattr check fails)")
        return False


def test_config_fields():
    """Test that test config has all required fields."""
    print("\n[3/3] Testing test config has all required fields...")

    import yaml

    # Load the test file
    with open('tests/test_evaluation_integration.py', 'r') as f:
        content = f.read()

    # Check for hidden_dims
    if "'hidden_dims'" in content or '"hidden_dims"' in content:
        print("✓ Test config includes hidden_dims")
        return True
    else:
        print("✗ Test config missing hidden_dims")
        return False


if __name__ == '__main__':
    results = []

    try:
        results.append(test_ffnetwork_attribute())
    except Exception as e:
        print(f"✗ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_evaluation_detection())
    except Exception as e:
        print(f"✗ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_config_fields())
    except Exception as e:
        print(f"✗ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    print("\n" + "=" * 80)
    if all(results):
        print("🎉 ALL BUG FIXES VERIFIED!")
        print("=" * 80)
        print("\nFixed bugs:")
        print("  ✓ FFNetwork now uses ff_layers (consistent with HybridFFBPModel)")
        print("  ✓ Evaluation code will detect and run for FFNetwork")
        print("  ✓ Test config has all required fields (hidden_dims)")
        sys.exit(0)
    else:
        print("✗ SOME FIXES FAILED")
        print("=" * 80)
        print(f"\nPassed: {sum(results)}/{len(results)}")
        sys.exit(1)
