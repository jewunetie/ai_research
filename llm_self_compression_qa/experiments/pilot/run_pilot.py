#!/usr/bin/env python3
"""
Pilot experiment script for testing the compression pipeline.

Runs a small-scale experiment on 10 documents to validate all components
work together before running the full experiment.

Usage:
    python experiments/pilot/run_pilot.py [--config CONFIG_FILE]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.openai_model import OpenAIModel
from src.data.loader import DataLoader
from src.data.validator import DocumentValidator
from src.compression.compressor import Compressor
from src.compression.prompts import SELF_COMPRESSION_PROMPT, HUMAN_READABLE_SUMMARY_PROMPT
from src.evaluation.supervisor import Supervisor
from src.evaluation.answerer import Answerer
from src.evaluation.metrics import MetricsCalculator
from src.baselines.full_context import FullContextBaseline
from src.baselines.no_context import NoContextBaseline
from src.evaluation.aggregator import ResultAggregator
from src.utils.logging_config import setup_experiment_logging
from src.utils.cost_tracker import CostTracker

from tqdm import tqdm


def run_pilot(
    num_documents: int = 10,
    num_questions: int = 5,
    output_dir: Path = None,
    model_name: str = "gpt-5.1-chat-latest",
):
    """
    Run pilot experiment.

    Args:
        num_documents: Number of documents to process
        num_questions: Questions per document
        output_dir: Output directory for results
        model_name: Model to use
    """
    # Setup
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/pilot/pilot_{timestamp}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_experiment_logging("pilot", output_dir)

    logger.info("Starting Pilot Experiment")
    logger.info(f"Documents: {num_documents}")
    logger.info(f"Questions per document: {num_questions}")
    logger.info(f"Model: {model_name}")

    # Initialize model
    logger.info("Initializing model...")
    try:
        model = OpenAIModel(
            model_name=model_name,
            temperature=0.0,
            seed=42,
            reasoning_effort="none",
        )
        logger.info(f"✓ Model initialized: {model.get_model_name()}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.info("Trying fallback model gpt-4o...")
        model = OpenAIModel(
            model_name="gpt-4o",
            temperature=0.0,
            seed=42,
        )
        logger.info(f"✓ Fallback model initialized: {model.get_model_name()}")

    # Initialize cost tracker
    cost_tracker = CostTracker()

    # Load data
    logger.info("Loading dataset...")
    loader = DataLoader(dataset_name="cnn_dailymail", seed=42)
    loader.load(split="test")
    documents = loader.sample(num_documents, shuffle=True)
    logger.info(f"✓ Loaded {len(documents)} documents")

    # Validate documents
    logger.info("Validating documents...")
    validator = DocumentValidator()
    validation_result = validator.validate_batch(documents)
    logger.info(
        f"✓ Validation: {validation_result['valid']}/{validation_result['total']} valid "
        f"({validation_result['valid_ratio']:.1%})"
    )

    documents = validation_result['valid_documents']

    # Initialize components
    logger.info("Initializing pipeline components...")
    compressor = Compressor(model, max_tokens=1500)
    supervisor = Supervisor(model)
    answerer = Answerer(model)
    metrics_calc = MetricsCalculator()
    full_context_baseline = FullContextBaseline(model)
    no_context_baseline = NoContextBaseline(model)

    aggregator = ResultAggregator()

    # Process documents
    logger.info(f"Processing {len(documents)} documents...")

    for doc_idx, doc in enumerate(tqdm(documents, desc="Processing documents")):
        doc_id = doc['id']
        logger.info(f"\n{'='*60}")
        logger.info(f"Document {doc_idx + 1}/{len(documents)}: {doc_id}")
        logger.info(f"{'='*60}")

        # 1. Compress document (self-compression variant)
        logger.info("Compressing document...")
        try:
            compressed_result = compressor.compress(
                doc['text'],
                prompt_template=SELF_COMPRESSION_PROMPT,
                max_tokens=1500,
            )
            logger.info(
                f"✓ Compressed: {compressed_result['compression_ratio']:.2f}x "
                f"({compressed_result['original_tokens']} → {compressed_result['token_count']} tokens)"
            )

            # Track cost
            cost_tracker.track_call(
                "compression",
                input_tokens=compressed_result['original_tokens'],
                output_tokens=compressed_result['token_count'],
            )

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            continue

        # 2. Generate questions and reference answers
        logger.info(f"Generating {num_questions} QA pairs...")
        try:
            qa_pairs = supervisor.generate_qa_pairs(
                doc['text'],
                num_questions=num_questions,
            )
            logger.info(f"✓ Generated {len(qa_pairs)} QA pairs")

            # Track cost
            cost_tracker.track_call(
                "qa_generation",
                input_tokens=model.count_tokens(doc['text']),
                output_tokens=500,  # Approximate
            )

        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            continue

        if len(qa_pairs) == 0:
            logger.warning("No QA pairs generated, skipping document")
            continue

        # 3. Answer questions from different contexts
        logger.info("Answering questions from different contexts...")

        for qa in qa_pairs:
            question = qa['question']
            reference = qa['reference_answer']

            # Answer from full context
            try:
                answer_full = full_context_baseline.answer(doc['text'], question)
                metrics_full = metrics_calc.compute_all_metrics(answer_full, reference)

                aggregator.add_result({
                    'doc_id': doc_id,
                    'question': question,
                    'reference_answer': reference,
                    'condition': 'full_context',
                    'answer': answer_full,
                    **metrics_full,
                })

                cost_tracker.track_call(
                    "qa_answering",
                    input_tokens=model.count_tokens(doc['text'] + question),
                    output_tokens=50,
                )

            except Exception as e:
                logger.error(f"Full context answering failed: {e}")

            # Answer from compressed context
            try:
                answer_compressed = answerer.answer(
                    compressed_result['compressed'],
                    question,
                )
                metrics_compressed = metrics_calc.compute_all_metrics(
                    answer_compressed,
                    reference,
                )

                aggregator.add_result({
                    'doc_id': doc_id,
                    'question': question,
                    'reference_answer': reference,
                    'condition': 'self_compression',
                    'answer': answer_compressed,
                    **metrics_compressed,
                })

                cost_tracker.track_call(
                    "qa_answering",
                    input_tokens=model.count_tokens(compressed_result['compressed'] + question),
                    output_tokens=50,
                )

            except Exception as e:
                logger.error(f"Compressed context answering failed: {e}")

            # Answer with no context
            try:
                answer_no_context = no_context_baseline.answer(question)
                metrics_no_context = metrics_calc.compute_all_metrics(
                    answer_no_context,
                    reference,
                )

                aggregator.add_result({
                    'doc_id': doc_id,
                    'question': question,
                    'reference_answer': reference,
                    'condition': 'no_context',
                    'answer': answer_no_context,
                    **metrics_no_context,
                })

                cost_tracker.track_call(
                    "qa_answering",
                    input_tokens=model.count_tokens(question),
                    output_tokens=50,
                )

            except Exception as e:
                logger.error(f"No context answering failed: {e}")

        logger.info(f"✓ Completed {len(qa_pairs)} questions")

    # Save results
    logger.info("\nSaving results...")

    # Summary statistics
    summary = aggregator.get_summary_statistics(group_by='condition')
    summary_path = output_dir / "summary_statistics.csv"
    summary.to_csv(summary_path)
    logger.info(f"✓ Summary statistics: {summary_path}")

    # Full results
    full_results_path = output_dir / "full_results.csv"
    aggregator.save_full_results(full_results_path)
    logger.info(f"✓ Full results: {full_results_path}")

    # Cost summary
    cost_summary = cost_tracker.get_summary()
    cost_path = output_dir / "cost_summary.json"
    with open(cost_path, 'w') as f:
        json.dump(cost_summary, f, indent=2)
    logger.info(f"✓ Cost summary: {cost_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PILOT EXPERIMENT COMPLETE")
    logger.info("="*80)

    logger.info("\nSummary Statistics by Condition:")
    logger.info("\n" + str(summary))

    logger.info(f"\nCost Summary:")
    logger.info(f"  Total API calls: {cost_summary['calls_made']:,}")
    logger.info(f"  Total tokens: {cost_summary['total_input_tokens'] + cost_summary['total_output_tokens']:,}")
    logger.info(f"  Estimated cost: ${cost_summary['estimated_cost']:.2f}")

    logger.info(f"\nResults saved to: {output_dir}")

    return aggregator, summary, cost_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run pilot experiment")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of documents to process (default: 10)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Questions per document (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1-chat-latest",
        help="Model to use (default: gpt-5.1-chat-latest)",
    )

    args = parser.parse_args()

    run_pilot(
        num_documents=args.num_docs,
        num_questions=args.num_questions,
        output_dir=args.output_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
