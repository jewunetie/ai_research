"""
Evaluation script for UNKNOWN Token Sink.

Evaluates trained model on multiple datasets:
- In-distribution (FineWeb-Edu test)
- Synthetic gibberish
- OOD datasets (SQuAD 2.0, PubMedQA, TruthfulQA)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from tqdm import tqdm

from models.unknown_token_model import UnknownTokenModel
from data.dataset import EvaluationDataset
from evaluation.metrics import (
    compute_unknown_rate,
    compute_perplexity,
    compute_next_token_accuracy,
    evaluate_abstention_classification,
    compute_confidence_metrics,
    MetricsTracker
)


def generate_predictions(
    model_wrapper: UnknownTokenModel,
    texts: List[str],
    max_new_tokens: int = 50,
    batch_size: int = 32,
    device: str = "cuda"
) -> List[str]:
    """
    Generate predictions for a list of texts.

    Args:
        model_wrapper: Model wrapper
        texts: List of input texts
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        device: Device to use

    Returns:
        List of generated text continuations
    """
    model_wrapper.model.eval()
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = model_wrapper.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model_wrapper.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=model_wrapper.tokenizer.pad_token_id
            )

        # Decode
        generated_texts = model_wrapper.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=False
        )

        predictions.extend(generated_texts)

    return predictions


def evaluate_dataset(
    model_wrapper: UnknownTokenModel,
    dataset_path: Path,
    dataset_name: str,
    eval_config: dict,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on a single dataset.

    Args:
        model_wrapper: Model wrapper
        dataset_path: Path to evaluation dataset
        dataset_name: Name of dataset for logging
        eval_config: Evaluation configuration
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = EvaluationDataset(
        data_path=dataset_path,
        tokenizer=model_wrapper.tokenizer,
        max_length=512
    )

    # Extract texts and labels
    texts = [ex['text'] for ex in dataset.examples]
    is_gibberish_labels = [ex['is_gibberish'] for ex in dataset.examples]

    print(f"Dataset size: {len(texts):,}")
    print(f"Gibberish examples: {sum(is_gibberish_labels):,}")
    print(f"Real examples: {sum(not x for x in is_gibberish_labels):,}\n")

    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(
        model_wrapper=model_wrapper,
        texts=texts,
        max_new_tokens=eval_config.get('max_new_tokens', 50),
        batch_size=eval_config.get('batch_size', 32),
        device=device
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {}

    # UNKNOWN rate
    unknown_rate = compute_unknown_rate(
        predictions,
        unknown_token=model_wrapper.unknown_token
    )
    metrics['unknown_rate'] = unknown_rate
    print(f"  UNKNOWN rate: {unknown_rate:.2%}")

    # Classification metrics (if we have gibberish labels)
    if any(is_gibberish_labels):
        classification_metrics = evaluate_abstention_classification(
            predictions,
            is_gibberish_labels,
            unknown_token=model_wrapper.unknown_token
        )
        metrics['classification'] = classification_metrics
        print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        print(f"  F1: {classification_metrics['f1']:.4f}")
        print(f"  Precision: {classification_metrics['precision']:.4f}")
        print(f"  Recall: {classification_metrics['recall']:.4f}")

    # Perplexity (only on non-gibberish texts)
    real_texts = [t for t, is_gib in zip(texts, is_gibberish_labels) if not is_gib]
    if real_texts:
        print("  Computing perplexity...")
        perplexity = compute_perplexity(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            texts=real_texts[:1000],  # Sample for speed
            device=device
        )
        metrics['perplexity'] = perplexity
        print(f"  Perplexity: {perplexity:.2f}")

    # Accuracy (only on non-gibberish texts)
    if real_texts:
        print("  Computing accuracy...")
        accuracy = compute_next_token_accuracy(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            texts=real_texts[:1000],  # Sample for speed
            device=device
        )
        metrics['accuracy'] = accuracy
        print(f"  Next-token accuracy: {accuracy:.4f}")

    # Confidence metrics
    print("  Computing confidence metrics...")
    confidence_metrics = compute_confidence_metrics(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        texts=texts[:500],  # Sample for speed
        device=device
    )
    metrics['confidence'] = confidence_metrics
    print(f"  Mean max prob: {confidence_metrics['mean_max_prob']:.4f}")

    return metrics


def check_success_criteria(metrics_tracker: MetricsTracker, targets: dict):
    """
    Check if model meets success criteria.

    Args:
        metrics_tracker: Tracker with all evaluation metrics
        targets: Target metrics from config
    """
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)

    all_passed = True
    results = []

    # In-distribution criteria
    if 'in_distribution' in metrics_tracker.metrics:
        in_dist = metrics_tracker.metrics['in_distribution']
        target = targets.get('in_distribution', {})

        # Check UNKNOWN rate
        unknown_rate = in_dist.get('unknown_rate', 1.0)
        max_allowed = target.get('unknown_rate_max', 0.05)
        passed = unknown_rate <= max_allowed
        all_passed &= passed
        results.append({
            'test': 'In-dist UNKNOWN rate',
            'value': f"{unknown_rate:.2%}",
            'target': f"<{max_allowed:.0%}",
            'passed': passed
        })

    # Synthetic gibberish criteria
    if 'synthetic_gibberish' in metrics_tracker.metrics:
        gibberish = metrics_tracker.metrics['synthetic_gibberish']
        target = targets.get('synthetic_gibberish', {})

        # Check UNKNOWN rate
        unknown_rate = gibberish.get('unknown_rate', 0.0)
        min_required = target.get('unknown_rate_min', 0.90)
        passed = unknown_rate >= min_required
        all_passed &= passed
        results.append({
            'test': 'Gibberish UNKNOWN rate',
            'value': f"{unknown_rate:.2%}",
            'target': f">{min_required:.0%}",
            'passed': passed
        })

    # Print results
    print()
    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{status} | {result['test']}: {result['value']} (target: {result['target']})")

    print("\n" + "="*60)
    if all_passed:
        print("✓ All success criteria met!")
    else:
        print("✗ Some criteria not met. Further tuning may be needed.")
    print("="*60)


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("UNKNOWN TOKEN SINK - EVALUATION")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded\n")

    eval_config = config.get('evaluation', {})
    datasets_config = config.get('datasets', {})
    targets = config.get('targets', {})

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    print()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_wrapper = UnknownTokenModel()
    model_wrapper.load(model_path)

    print()

    # Initialize metrics tracker
    tracker = MetricsTracker()

    # Evaluate on each dataset
    data_dir = Path(args.data_dir)

    # 1. In-distribution test set
    in_dist_path = data_dir / "test" / "test_real.jsonl"
    if in_dist_path.exists():
        metrics = evaluate_dataset(
            model_wrapper=model_wrapper,
            dataset_path=in_dist_path,
            dataset_name="In-Distribution (FineWeb-Edu Test)",
            eval_config=eval_config,
            device=device
        )
        tracker.add_metrics('in_distribution', metrics)

    # 2. Synthetic gibberish
    gibberish_path = data_dir / "test" / "gibberish_synthetic.jsonl"
    if gibberish_path.exists():
        metrics = evaluate_dataset(
            model_wrapper=model_wrapper,
            dataset_path=gibberish_path,
            dataset_name="Synthetic Gibberish",
            eval_config=eval_config,
            device=device
        )
        tracker.add_metrics('synthetic_gibberish', metrics)

    # Print summary
    tracker.print_summary()

    # Check success criteria
    check_success_criteria(tracker, targets)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "evaluation_results.json"
        tracker.save_to_file(results_file)

    print()
    print("="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNKNOWN Token Sink model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/eval_config.yaml",
        help="Path to evaluation config YAML"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for output results"
    )

    args = parser.parse_args()
    main(args)
