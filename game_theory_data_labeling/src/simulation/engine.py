"""Simulation engine for running multiple game instances."""

from dataclasses import dataclass
import numpy as np

from src.simulation.game import Game, GameInstance, ExperimentConfig
from src.evaluation.metrics import MetricResult, evaluate_game


@dataclass
class ExperimentResults:
    """Aggregated results across multiple runs."""

    config: ExperimentConfig
    metrics: list[MetricResult]  # One per run
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]  # 95% CI


class SimulationEngine:
    """Orchestrates multiple game runs and collects results."""

    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: Experiment configuration
        """
        self.config = config

    def run_experiment(self) -> ExperimentResults:
        """Run multiple games and aggregate results.

        Returns:
            ExperimentResults with aggregated statistics
        """
        metrics = []

        for run_id in range(self.config.num_runs):
            # Create RNG with seed for reproducibility
            seed = self.config.random_seed + run_id
            rng = np.random.default_rng(seed)

            # Create and run game
            game = Game(self.config, rng)
            game_instance = game.run()

            # Evaluate
            metric = evaluate_game(
                game_instance.result.aggregated_labels,
                game_instance.true_labels,
                game_instance.result.payments,
            )
            metrics.append(metric)

            # Log progress
            if (run_id + 1) % max(1, self.config.num_runs // 10) == 0 or run_id == 0:
                print(
                    f"Completed {run_id + 1}/{self.config.num_runs} runs "
                    f"(accuracy: {metric.accuracy:.3f})"
                )

        # Aggregate results
        return self._aggregate_results(metrics)

    def _aggregate_results(
        self, metrics: list[MetricResult]
    ) -> ExperimentResults:
        """Aggregate metrics across runs.

        Args:
            metrics: List of metrics from each run

        Returns:
            ExperimentResults with aggregated statistics
        """
        # Convert to arrays
        metric_arrays = {
            "accuracy": np.array([m.accuracy for m in metrics]),
            "f1_score": np.array([m.f1_score for m in metrics]),
            "precision": np.array([m.precision for m in metrics]),
            "recall": np.array([m.recall for m in metrics]),
            "total_payment": np.array([m.total_payment for m in metrics]),
            "average_payment": np.array([m.average_payment for m in metrics]),
            "payment_std": np.array([m.payment_std for m in metrics]),
            "quality_per_dollar": np.array([m.quality_per_dollar for m in metrics]),
        }

        # Compute means
        mean_metrics = {field: float(np.mean(values)) for field, values in metric_arrays.items()}

        # Compute std
        std_metrics = {field: float(np.std(values)) for field, values in metric_arrays.items()}

        # Compute 95% confidence intervals
        confidence_intervals = {}
        for field, values in metric_arrays.items():
            mean = np.mean(values)
            sem = np.std(values) / np.sqrt(len(values))
            ci = (float(mean - 1.96 * sem), float(mean + 1.96 * sem))
            confidence_intervals[field] = ci

        return ExperimentResults(
            config=self.config,
            metrics=metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals,
        )
