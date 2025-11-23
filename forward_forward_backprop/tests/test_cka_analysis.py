"""
Test CKA (Centered Kernel Alignment) similarity analysis.

This test verifies:
1. CKA computation is mathematically correct
2. Identical features give CKA = 1.0
3. Different features give CKA < 1.0
4. CKA is invariant to orthogonal transformations
5. compare_model_representations works with real models
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_cka, compare_model_representations, extract_features
from src.models.mlp import HybridFFBPModel

print("=" * 80)
print("CKA SIMILARITY ANALYSIS TEST")
print("=" * 80)


def test_cka_identical_features():
    """Test that CKA of identical features is 1.0."""
    print("\n[Test 1/5] CKA of identical features...")

    # Create random features
    torch.manual_seed(42)
    features = torch.randn(100, 50)

    # CKA of identical features should be 1.0
    cka_score = compute_cka(features, features)

    print(f"  CKA(X, X) = {cka_score:.6f}")

    if abs(cka_score - 1.0) < 1e-5:
        print("  ✓ CKA of identical features is 1.0")
        return True
    else:
        print(f"  ✗ Expected 1.0, got {cka_score:.6f}")
        return False


def test_cka_different_features():
    """Test that CKA of different features is < 1.0."""
    print("\n[Test 2/5] CKA of different features...")

    torch.manual_seed(42)
    features_x = torch.randn(100, 50)

    torch.manual_seed(123)
    features_y = torch.randn(100, 50)

    # CKA of different random features should be < 1.0
    cka_score = compute_cka(features_x, features_y)

    print(f"  CKA(X, Y) = {cka_score:.6f}")

    if 0.0 <= cka_score < 1.0:
        print("  ✓ CKA of different features is in [0, 1) range")
        return True
    else:
        print(f"  ✗ Expected 0 <= score < 1, got {cka_score:.6f}")
        return False


def test_cka_orthogonal_invariance():
    """Test that CKA is invariant to orthogonal transformations."""
    print("\n[Test 3/5] CKA orthogonal invariance...")

    torch.manual_seed(42)
    features_x = torch.randn(100, 50)
    features_y = torch.randn(100, 50)

    # Compute CKA
    cka_original = compute_cka(features_x, features_y)

    # Create orthogonal transformation (random rotation)
    # Use QR decomposition to get orthogonal matrix
    random_matrix = torch.randn(50, 50)
    Q, _ = torch.linalg.qr(random_matrix)

    # Transform features_y
    features_y_rotated = features_y @ Q

    # Compute CKA with rotated features
    cka_rotated = compute_cka(features_x, features_y_rotated)

    print(f"  CKA(X, Y) = {cka_original:.6f}")
    print(f"  CKA(X, QY) = {cka_rotated:.6f}")
    print(f"  Difference = {abs(cka_original - cka_rotated):.8f}")

    # Should be approximately equal (allowing small numerical errors)
    if abs(cka_original - cka_rotated) < 1e-4:
        print("  ✓ CKA is invariant to orthogonal transformations")
        return True
    else:
        print(f"  ✗ CKA changed by {abs(cka_original - cka_rotated):.8f}")
        return False


def test_cka_dimension_mismatch():
    """Test that CKA handles features with different dimensions."""
    print("\n[Test 4/5] CKA with different feature dimensions...")

    torch.manual_seed(42)
    features_x = torch.randn(100, 30)  # 30-dim features
    features_y = torch.randn(100, 50)  # 50-dim features

    # Should work fine - CKA doesn't require same dimensions
    cka_score = compute_cka(features_x, features_y)

    print(f"  Features X: {features_x.shape}")
    print(f"  Features Y: {features_y.shape}")
    print(f"  CKA = {cka_score:.6f}")

    if 0.0 <= cka_score <= 1.0:
        print("  ✓ CKA handles different feature dimensions")
        return True
    else:
        print(f"  ✗ CKA out of range: {cka_score:.6f}")
        return False


def test_compare_model_representations():
    """Test comparing representations between two models."""
    print("\n[Test 5/5] Comparing model representations...")

    # Create two models with same architecture but different initializations
    torch.manual_seed(42)
    model1 = HybridFFBPModel(
        input_dim=784,
        ff_hidden_dims=[500, 500],
        num_classes=10,
        ff_threshold=2.0
    )

    torch.manual_seed(123)
    model2 = HybridFFBPModel(
        input_dim=784,
        ff_hidden_dims=[500, 500],
        num_classes=10,
        ff_threshold=2.0
    )

    # Create dummy data
    torch.manual_seed(0)
    dummy_images = torch.randn(200, 784)
    dummy_labels = torch.randint(0, 10, (200,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=50)

    # Compare representations
    device = torch.device('cpu')
    model1.eval()
    model2.eval()

    try:
        similarities = compare_model_representations(
            model1, model2, loader, device,
            layer_indices=[0, 1]  # Compare first two layers
        )

        print(f"\n  Results:")
        for layer, score in similarities.items():
            print(f"    {layer}: {score:.4f}")

        # Verify all scores are in valid range
        all_valid = all(0.0 <= score <= 1.0 for score in similarities.values())

        if all_valid and len(similarities) == 2:
            print("  ✓ Model comparison works correctly")
            return True
        else:
            print("  ✗ Invalid CKA scores or wrong number of layers")
            return False

    except Exception as e:
        print(f"  ✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cka_with_real_model_features():
    """Bonus test: Extract features from real model and compute CKA."""
    print("\n[Bonus Test] CKA with real model features...")

    # Create model
    torch.manual_seed(42)
    model = HybridFFBPModel(
        input_dim=784,
        ff_hidden_dims=[500, 500],
        num_classes=10,
        ff_threshold=2.0
    )
    model.eval()

    # Create dummy data
    torch.manual_seed(0)
    dummy_images = torch.randn(100, 784)
    dummy_labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=50)

    device = torch.device('cpu')

    try:
        # Extract features from layer 0
        features_layer0, _ = extract_features(model, loader, device, layer_index=0)

        # Extract features from layer 1
        features_layer1, _ = extract_features(model, loader, device, layer_index=1)

        # Compute CKA between different layers
        cka_score = compute_cka(features_layer0, features_layer1)

        print(f"  Layer 0 features: {features_layer0.shape}")
        print(f"  Layer 1 features: {features_layer1.shape}")
        print(f"  CKA(layer0, layer1) = {cka_score:.4f}")

        if 0.0 <= cka_score <= 1.0:
            print("  ✓ Real model feature extraction and CKA works")
            return True
        else:
            print(f"  ✗ CKA out of range: {cka_score:.4f}")
            return False

    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []

    # Run all tests
    results.append(test_cka_identical_features())
    results.append(test_cka_different_features())
    results.append(test_cka_orthogonal_invariance())
    results.append(test_cka_dimension_mismatch())
    results.append(test_compare_model_representations())
    results.append(test_cka_with_real_model_features())

    # Summary
    num_passed = sum(results)
    num_total = len(results)

    print("\n" + "=" * 80)
    if num_passed == num_total:
        print("🎉 ALL CKA TESTS PASSED!")
        print("=" * 80)
        print("\nCKA similarity analysis is working correctly:")
        print("  ✓ Identical features give CKA = 1.0")
        print("  ✓ Different features give CKA in [0, 1)")
        print("  ✓ CKA is invariant to orthogonal transformations")
        print("  ✓ Handles different feature dimensions")
        print("  ✓ Model comparison works with real models")
        print("  ✓ Feature extraction and CKA computation works")
        sys.exit(0)
    else:
        print(f"✗ CKA TESTS FAILED ({num_passed}/{num_total} passed)")
        print("=" * 80)
        sys.exit(1)
