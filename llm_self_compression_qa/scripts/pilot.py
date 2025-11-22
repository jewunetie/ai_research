#!/usr/bin/env python3
"""
Pilot experiment script for LLM self-compression QA research.

Runs a small-scale experiment (10 documents) to validate the pipeline
and get initial results before scaling up.

Usage:
    python scripts/pilot.py

Results are saved to: results/pilot/
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.openai_model import OpenAIModel
from src.data.loader import DataLoader
from src.compression.compressor import Compressor
from src.evaluation.supervisor import Supervisor
from src.evaluation.answerer import Answerer
from src.evaluation.metrics import compute_all_metrics
from src.baselines.full_context import FullContextBaseline
from src.baselines.no_context import NoContextBaseline

# Load environment
load_dotenv()


def main():
    """Run pilot experiment."""
    print("=" * 80)
    print("LLM Self-Compression QA - Pilot Experiment")
    print("=" * 80)
    print()

    # Configuration
    NUM_DOCS = 10
    NUM_QUESTIONS_PER_DOC = 10
    TOKEN_LIMIT = 1500
    MODEL_NAME = os.getenv("DEFAULT_MODEL", "gpt-5.1-chat-latest")

    print(f"Configuration:")
    print(f"  Documents: {NUM_DOCS}")
    print(f"  Questions per doc: {NUM_QUESTIONS_PER_DOC}")
    print(f"  Token limit: {TOKEN_LIMIT}")
    print(f"  Model: {MODEL_NAME}")
    print()

    # Create results directory
    results_dir = Path("results/pilot")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize model
    print("Initializing model...")
    try:
        model = OpenAIModel(
            model_name=MODEL_NAME,
            temperature=0.0,
            seed=42,
        )
        print(f"✓ Model initialized: {model.get_model_name()}")

        # Test API connection
        test_response = model.generate("Say 'OK' if you can read this.")
        print(f"✓ API test successful: {test_response}")
        print(f"  API type: {model.get_api_type()}")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        print()
        print("Please run verify_api.py first to check your setup.")
        return 1

    print()

    # Load data
    print("Loading dataset...")
    loader = DataLoader(dataset_name="cnn_dailymail", version="3.0.0", seed=42)
    loader.load(split="test")
    documents = loader.sample(NUM_DOCS, shuffle=True)
    print(f"✓ Loaded {len(documents)} documents")
    print()

    # Initialize components
    print("Initializing pipeline components...")
    compressor = Compressor(model, max_tokens=TOKEN_LIMIT)
    supervisor = Supervisor(model)
    answerer = Answerer(model)
    full_context_baseline = FullContextBaseline(model)
    no_context_baseline = NoContextBaseline()
    print("✓ All components initialized")
    print()

    # Run experiment
    results = []

    for i, doc in enumerate(documents):
        print("-" * 80)
        print(f"Document {i+1}/{NUM_DOCS}: {doc['id']}")
        print("-" * 80)

        text = doc["text"]
        doc_id = doc["id"]

        # Count original tokens
        original_tokens = model.count_tokens(text)
        print(f"Original text: {original_tokens} tokens")

        try:
            # Step 1: Generate questions (Supervisor with full context)
            print(f"Generating {NUM_QUESTIONS_PER_DOC} questions...")
            qa_pairs = supervisor.generate_questions(text, num_questions=NUM_QUESTIONS_PER_DOC)
            print(f"✓ Generated {len(qa_pairs)} questions")

            # Step 2: Compress document (Self-compression)
            print("Compressing document (self-compression)...")
            compressed, compress_meta = compressor.compress(text, compression_type="self")
            print(f"✓ Compressed to {compress_meta['compressed_tokens']} tokens "
                  f"(ratio: {compress_meta['compression_ratio']:.2f}x)")

            # Step 3: Answer questions from compressed context
            print("Answering questions from compressed context...")
            compressed_answers = []
            for j, qa in enumerate(qa_pairs):
                answer = answerer.answer_question(qa["question"], compressed)
                compressed_answers.append(answer)
                print(f"  Q{j+1}: {qa['question'][:60]}...")

            # Step 4: Baselines
            print("Running baselines...")

            # Full context baseline
            full_text, full_meta = full_context_baseline.process(text)
            full_answers = []
            for qa in qa_pairs:
                answer = answerer.answer_question(qa["question"], full_text)
                full_answers.append(answer)

            # No context baseline
            no_text, no_meta = no_context_baseline.process(text)
            no_answers = []
            for qa in qa_pairs:
                answer = answerer.answer_question(qa["question"], no_text)
                no_answers.append(answer)

            print("✓ Baselines complete")

            # Step 5: Compute metrics
            print("Computing metrics...")
            doc_results = {
                "doc_id": doc_id,
                "original_tokens": original_tokens,
                "compression_metadata": compress_meta,
                "questions": [],
            }

            for j, qa in enumerate(qa_pairs):
                question = qa["question"]
                reference = qa["reference_answer"]

                # Metrics for compressed context
                compressed_metrics = compute_all_metrics(compressed_answers[j], reference)

                # Metrics for full context
                full_metrics = compute_all_metrics(full_answers[j], reference)

                # Metrics for no context
                no_metrics = compute_all_metrics(no_answers[j], reference)

                doc_results["questions"].append({
                    "question": question,
                    "reference_answer": reference,
                    "compressed_answer": compressed_answers[j],
                    "compressed_metrics": compressed_metrics,
                    "full_context_answer": full_answers[j],
                    "full_context_metrics": full_metrics,
                    "no_context_answer": no_answers[j],
                    "no_context_metrics": no_metrics,
                })

            results.append(doc_results)
            print(f"✓ Document {i+1} complete")
            print()

        except Exception as e:
            print(f"✗ Error processing document {doc_id}: {e}")
            continue

    # Save results
    print("=" * 80)
    print("Saving results...")
    output_file = results_dir / f"pilot_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "num_docs": NUM_DOCS,
                "num_questions_per_doc": NUM_QUESTIONS_PER_DOC,
                "token_limit": TOKEN_LIMIT,
                "model": MODEL_NAME,
                "timestamp": timestamp,
            },
            "results": results,
        }, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print()

    # Compute aggregate statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    total_questions = sum(len(doc["questions"]) for doc in results)
    print(f"Total documents: {len(results)}")
    print(f"Total questions: {total_questions}")
    print()

    # Average metrics across all questions
    compressed_em = []
    compressed_f1 = []
    full_em = []
    full_f1 = []
    no_em = []
    no_f1 = []

    for doc in results:
        for q in doc["questions"]:
            compressed_em.append(q["compressed_metrics"]["exact_match"])
            compressed_f1.append(q["compressed_metrics"]["f1"])
            full_em.append(q["full_context_metrics"]["exact_match"])
            full_f1.append(q["full_context_metrics"]["f1"])
            no_em.append(q["no_context_metrics"]["exact_match"])
            no_f1.append(q["no_context_metrics"]["f1"])

    print("Average Exact Match:")
    print(f"  Self-Compression:  {sum(compressed_em)/len(compressed_em):.3f}")
    print(f"  Full Context:      {sum(full_em)/len(full_em):.3f}")
    print(f"  No Context:        {sum(no_em)/len(no_em):.3f}")
    print()

    print("Average F1 Score:")
    print(f"  Self-Compression:  {sum(compressed_f1)/len(compressed_f1):.3f}")
    print(f"  Full Context:      {sum(full_f1)/len(full_f1):.3f}")
    print(f"  No Context:        {sum(no_f1)/len(no_f1):.3f}")
    print()

    # Compression stats
    compression_ratios = [doc["compression_metadata"]["compression_ratio"] for doc in results]
    avg_ratio = sum(compression_ratios) / len(compression_ratios)
    print(f"Average Compression Ratio: {avg_ratio:.2f}x")
    print()

    print("=" * 80)
    print("Pilot experiment complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
