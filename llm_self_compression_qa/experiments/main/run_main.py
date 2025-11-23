#!/usr/bin/env python3
"""
Main experiment runner for LLM self-compression QA research.

Runs comprehensive experiments on 100 documents with multiple compression
variants and baselines, featuring progress tracking and cost estimation.

Usage:
    python experiments/main/run_main.py [--config CONFIG_PATH] [--resume]
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import ExperimentConfig
from src.utils.cost_tracker import CostTracker
from src.models.openai_model import OpenAIModel
from src.data.loader import DataLoader
from src.compression.compressor import Compressor
from src.compression.prompts import (
    SELF_COMPRESSION_PROMPT,
    HUMAN_READABLE_SUMMARY_PROMPT,
)
from src.evaluation.supervisor import Supervisor
from src.evaluation.answerer import Answerer
from src.evaluation.metrics import compute_all_metrics
from src.baselines.full_context import FullContextBaseline
from src.baselines.no_context import NoContextBaseline
from src.baselines.random_tokens import RandomTokenBaseline

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠  Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars")

# Load environment
load_dotenv()


# Prompt registry
PROMPT_REGISTRY = {
    "SELF_COMPRESSION_PROMPT": SELF_COMPRESSION_PROMPT,
    "HUMAN_READABLE_SUMMARY_PROMPT": HUMAN_READABLE_SUMMARY_PROMPT,
}


class MainExperiment:
    """Main experiment runner with progress tracking and checkpointing."""

    def __init__(self, config: ExperimentConfig, resume: bool = False):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
            resume: Whether to resume from checkpoint
        """
        self.config = config
        self.resume = resume
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("=" * 80)
        print(f"Main Experiment: {config.name}")
        print("=" * 80)
        print()

        print("Initializing model...")
        self.model = OpenAIModel(
            model_name=config.model.name,
            temperature=config.model.temperature,
            seed=config.model.seed,
            reasoning_effort=config.model.reasoning_effort,
        )
        print(f"✓ Model: {self.model.get_model_name()}")
        print(f"  API type: {self.model.get_api_type()}")
        print()

        # Initialize cost tracker
        if config.cost_tracking_enabled:
            self.cost_tracker = CostTracker(
                config.pricing,
                config.model.name
            )
        else:
            self.cost_tracker = None

        # Initialize components
        print("Initializing components...")
        self.compressor = Compressor(self.model, max_tokens=config.token_limit)
        self.supervisor = Supervisor(self.model)
        self.answerer = Answerer(self.model)
        print("✓ All components ready")
        print()

        # Initialize baselines
        self.baselines = {}
        if "full_context" in config.baselines_enabled:
            self.baselines["full_context"] = FullContextBaseline(self.model)
        if "no_context" in config.baselines_enabled:
            self.baselines["no_context"] = NoContextBaseline()
        if "random_tokens" in config.baselines_enabled:
            self.baselines["random_tokens"] = RandomTokenBaseline(
                self.model, max_tokens=config.token_limit
            )

        if self.baselines:
            print(f"✓ Baselines: {', '.join(self.baselines.keys())}")
        else:
            print("⚠  No baselines enabled")
        print()

        # Load data
        print("Loading dataset...")
        self.loader = DataLoader(
            dataset_name=config.dataset,
            seed=config.seed
        )
        self.loader.load(split=config.dataset_split)
        self.documents = self.loader.sample(config.num_documents, shuffle=True)
        print(f"✓ Loaded {len(self.documents)} documents")
        print()

        # Results storage
        self.results = []
        self.failed_docs = []

        # Load checkpoint if resuming
        if resume:
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load results from previous run to resume."""
        checkpoint_file = self.results_dir / f"checkpoint_{self.config.name}.json"
        if checkpoint_file.exists():
            print("Loading checkpoint...")
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            self.results = checkpoint.get("results", [])
            self.failed_docs = checkpoint.get("failed_documents", [])
            print(f"✓ Resumed from checkpoint: {len(self.results)} documents completed")
            print()
        else:
            print("⚠  No checkpoint found, starting from beginning")
            print()

    def _save_checkpoint(self):
        """Save intermediate results as checkpoint."""
        checkpoint_file = self.results_dir / f"checkpoint_{self.config.name}.json"
        with open(checkpoint_file, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "results": self.results,
                "failed_documents": self.failed_docs,
                "timestamp": datetime.now().isoformat(),
                "completed": len(self.results),
                "total": self.config.num_documents,
            }, f, indent=2)

    def _get_completed_doc_ids(self) -> set:
        """Get set of document IDs already processed."""
        return {doc["doc_id"] for doc in self.results}

    def _track_api_call(self, prompt: str, response: str):
        """Track an API call for cost estimation."""
        if self.cost_tracker:
            input_tokens = self.model.count_tokens(prompt)
            output_tokens = self.model.count_tokens(response)
            self.cost_tracker.track_call(input_tokens, output_tokens)

    def run(self):
        """Run the complete experiment."""
        start_time = datetime.now()

        print("=" * 80)
        print("Starting Experiment")
        print("=" * 80)
        print(f"Documents: {self.config.num_documents}")
        print(f"Questions per doc: {self.config.questions_per_document}")
        print(f"Compression variants: {len(self.config.compression_variants)}")
        print(f"Baselines: {len(self.baselines)}")
        print()

        # Validate configuration
        if len(self.config.compression_variants) == 0:
            raise ValueError("No compression variants specified in configuration. "
                           "Add at least one variant in the YAML config file.")

        # Get already completed docs
        completed_ids = self._get_completed_doc_ids()

        # Filter documents to process
        docs_to_process = [
            doc for doc in self.documents
            if doc["id"] not in completed_ids
        ]

        print(f"Documents to process: {len(docs_to_process)}")
        if len(completed_ids) > 0:
            print(f"Already completed: {len(completed_ids)}")
        print()

        # Create progress bar
        if TQDM_AVAILABLE and self.config.show_progress:
            pbar = tqdm(
                docs_to_process,
                desc="Processing documents",
                initial=len(completed_ids),
                total=self.config.num_documents
            )
        else:
            pbar = docs_to_process

        # Process each document
        for i, doc in enumerate(pbar):
            doc_start = datetime.now()

            if not TQDM_AVAILABLE:
                print("-" * 80)
                print(f"Document {len(completed_ids) + i + 1}/{self.config.num_documents}: {doc['id']}")
                print("-" * 80)

            try:
                result = self._process_document(doc)
                self.results.append(result)

                # Update progress bar with stats
                if TQDM_AVAILABLE and self.config.show_progress:
                    stats = {
                        "completed": len(self.results),
                        "failed": len(self.failed_docs)
                    }
                    if self.cost_tracker:
                        stats["cost"] = f"${self.cost_tracker.estimate.estimated_cost:.2f}"
                    pbar.set_postfix(stats)

                # Checkpoint periodically
                if (len(self.results) % self.config.checkpoint_interval) == 0:
                    self._save_checkpoint()

                doc_elapsed = (datetime.now() - doc_start).total_seconds()
                if self.config.verbose and not TQDM_AVAILABLE:
                    print(f"✓ Completed in {doc_elapsed:.1f}s")
                    print()

            except Exception as e:
                error_msg = str(e)
                self.failed_docs.append({
                    "doc_id": doc["id"],
                    "error": error_msg
                })
                if TQDM_AVAILABLE:
                    tqdm.write(f"✗ Error processing {doc['id']}: {error_msg}")
                else:
                    print(f"✗ Error processing {doc['id']}: {error_msg}")
                    print()

        if TQDM_AVAILABLE and self.config.show_progress:
            pbar.close()

        # Save final results
        self._save_final_results()

        # Print summary
        self._print_summary(start_time)

    def _process_document(self, doc: Dict) -> Dict:
        """Process a single document through all variants and baselines."""
        doc_id = doc["id"]
        text = doc["text"]
        original_tokens = self.model.count_tokens(text)

        # Step 1: Generate questions (once per document)
        qa_prompt = f"Based on the following text, generate {self.config.questions_per_document} diverse questions..."
        qa_pairs = self.supervisor.generate_questions(
            text,
            num_questions=self.config.questions_per_document
        )
        self._track_api_call(qa_prompt, str(qa_pairs))

        if len(qa_pairs) == 0:
            raise ValueError("No questions generated")

        # Store compressions for each variant
        compressions = {}

        # Step 2: Compress with each variant
        for variant in self.config.compression_variants:
            # Validate prompt key exists
            if variant.prompt_key not in PROMPT_REGISTRY:
                raise ValueError(
                    f"Unknown prompt_key '{variant.prompt_key}' in variant '{variant.name}'. "
                    f"Available keys: {list(PROMPT_REGISTRY.keys())}"
                )

            prompt_template = PROMPT_REGISTRY[variant.prompt_key]
            compressed, metadata = self.compressor.compress(
                text,
                compression_type="self" if "SELF" in variant.prompt_key else "human_readable"
            )
            self._track_api_call(prompt_template, compressed)

            compressions[variant.name] = {
                "compressed": compressed,
                "metadata": metadata
            }

        # Step 3: Answer questions from each compression variant
        variant_results = {}
        for variant_name, comp_data in compressions.items():
            compressed_text = comp_data["compressed"]
            answers = []

            for qa in qa_pairs:
                answer = self.answerer.answer_question(
                    qa["question"],
                    compressed_text
                )
                self._track_api_call(qa["question"], answer)

                metrics = compute_all_metrics(answer, qa["reference_answer"])

                answers.append({
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "answer": answer,
                    "metrics": metrics
                })

            variant_results[variant_name] = {
                "compression_metadata": comp_data["metadata"],
                "answers": answers
            }

        # Step 4: Run baselines
        baseline_results = {}
        for baseline_name, baseline in self.baselines.items():
            # Get baseline context (all baselines use same interface)
            context, metadata = baseline.process(text)

            # Answer questions
            answers = []
            for qa in qa_pairs:
                answer = self.answerer.answer_question(
                    qa["question"],
                    context
                )
                self._track_api_call(qa["question"], answer)

                metrics = compute_all_metrics(answer, qa["reference_answer"])

                answers.append({
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "answer": answer,
                    "metrics": metrics
                })

            baseline_results[baseline_name] = {
                "metadata": metadata,
                "answers": answers
            }

        # Compile document result
        return {
            "doc_id": doc_id,
            "original_tokens": original_tokens,
            "num_questions": len(qa_pairs),
            "variants": variant_results,
            "baselines": baseline_results,
        }

    def _save_final_results(self):
        """Save final experiment results."""
        output_file = self.results_dir / f"{self.config.name}_{self.timestamp}.json"

        results_data = {
            "config": self.config.to_dict(),
            "timestamp": self.timestamp,
            "results": self.results,
            "failed_documents": self.failed_docs,
            "summary": {
                "total_attempted": self.config.num_documents,
                "successful": len(self.results),
                "failed": len(self.failed_docs),
                "completion_rate": (len(self.results) / self.config.num_documents
                                   if self.config.num_documents > 0 else 0.0),
            }
        }

        # Add cost tracking if enabled
        if self.cost_tracker:
            results_data["cost_summary"] = self.cost_tracker.estimate.to_dict()

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"✓ Results saved to: {output_file}")

    def _print_summary(self, start_time: datetime):
        """Print experiment summary."""
        elapsed = (datetime.now() - start_time).total_seconds()

        print()
        print("=" * 80)
        print("Experiment Complete")
        print("=" * 80)
        print()
        print(f"Documents processed: {len(self.results)}/{self.config.num_documents}")
        print(f"Failed: {len(self.failed_docs)}")

        success_rate = (len(self.results) / self.config.num_documents * 100
                       if self.config.num_documents > 0 else 0.0)
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print()

        if self.cost_tracker:
            print(self.cost_tracker.get_summary())
            print()

        # Calculate average metrics
        if len(self.results) > 0:
            print("Average Metrics:")
            print()

            for variant in self.config.compression_variants:
                variant_name = variant.name
                all_f1 = []
                all_em = []

                for doc in self.results:
                    for ans in doc["variants"][variant_name]["answers"]:
                        all_f1.append(ans["metrics"]["f1"])
                        all_em.append(ans["metrics"]["exact_match"])

                if all_f1:
                    print(f"  {variant_name}:")
                    print(f"    F1: {sum(all_f1)/len(all_f1):.3f}")
                    print(f"    EM: {sum(all_em)/len(all_em):.3f}")

            print()

            for baseline_name in self.baselines.keys():
                all_f1 = []
                all_em = []

                for doc in self.results:
                    for ans in doc["baselines"][baseline_name]["answers"]:
                        all_f1.append(ans["metrics"]["f1"])
                        all_em.append(ans["metrics"]["exact_match"])

                if all_f1:
                    print(f"  {baseline_name}:")
                    print(f"    F1: {sum(all_f1)/len(all_f1):.3f}")
                    print(f"    EM: {sum(all_em)/len(all_em):.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run main LLM self-compression QA experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/main_config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ExperimentConfig.from_yaml(args.config)
    print(f"✓ Configuration loaded: {config.name}")
    print()

    # Run experiment
    experiment = MainExperiment(config, resume=args.resume)
    experiment.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
