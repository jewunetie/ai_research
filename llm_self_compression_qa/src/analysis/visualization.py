"""Visualization utilities for experiment results.

Provides publication-quality plots for analyzing compression experiment results.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings


class ExperimentVisualizer:
    """
    Create publication-quality visualizations for compression experiments.

    Supports:
    - Performance comparison bar charts
    - Score distribution box plots
    - Compression ratio vs performance scatter plots
    - Performance by question type heatmaps
    - Degradation curves
    - Statistical comparison plots
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-paper",
        context: str = "paper",
        palette: str = "Set2",
        fig_size: Tuple[int, int] = (10, 6),
        dpi: int = 300
    ):
        """
        Initialize visualizer with style settings.

        Args:
            style: Matplotlib style ('seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid', etc.)
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
            palette: Color palette ('Set2', 'husl', 'deep', etc.)
            fig_size: Default figure size (width, height)
            dpi: Resolution for saved figures
        """
        # Try to set style, fallback to defaults if not available
        try:
            plt.style.use(style)
        except:
            # Fallback to basic settings
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.grid': True,
                'grid.alpha': 0.3,
            })

        sns.set_context(context)
        sns.set_palette(palette)

        self.default_fig_size = fig_size
        self.dpi = dpi

        # Style settings for publication-quality
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 13,
            'lines.linewidth': 2,
            'lines.markersize': 6,
        })

    def plot_performance_comparison(
        self,
        df: pd.DataFrame,
        metric: str = "f1",
        condition_col: str = "condition",
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        show_error_bars: bool = True,
        show_counts: bool = True
    ) -> plt.Figure:
        """
        Bar chart comparing performance across conditions.

        Args:
            df: DataFrame with results
            metric: Metric to plot
            condition_col: Column with condition names
            save_path: Path to save figure (optional)
            title: Plot title (optional)
            show_error_bars: Show standard error bars
            show_counts: Show sample counts on bars

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.default_fig_size)

        # Compute statistics
        stats = df.groupby(condition_col)[metric].agg(['mean', 'std', 'count', 'sem'])
        stats = stats.sort_values('mean', ascending=False)

        # Create bar plot
        x_pos = np.arange(len(stats))
        bars = ax.bar(
            x_pos,
            stats['mean'],
            yerr=stats['sem'] if show_error_bars else None,
            capsize=5,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )

        # Customize
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_title(title or f'Performance Comparison: {metric.upper()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats.index, rotation=45, ha='right')
        ax.set_ylim(0, min(1.0, stats['mean'].max() * 1.15))

        # Add count annotations
        if show_counts:
            for i, (bar, count) in enumerate(zip(bars, stats['count'])):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'n={int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='gray'
                )

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_score_distributions(
        self,
        df: pd.DataFrame,
        metric: str = "f1",
        condition_col: str = "condition",
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        plot_type: str = "box"  # "box", "violin", or "both"
    ) -> plt.Figure:
        """
        Box/violin plot showing score distributions.

        Args:
            df: DataFrame with results
            metric: Metric to plot
            condition_col: Column with condition names
            save_path: Path to save figure
            title: Plot title
            plot_type: "box", "violin", or "both"

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.default_fig_size)

        # Sort conditions by median
        condition_order = df.groupby(condition_col)[metric].median().sort_values(ascending=False).index

        if plot_type == "box":
            sns.boxplot(
                data=df,
                x=condition_col,
                y=metric,
                order=condition_order,
                ax=ax,
                palette="Set2",
                linewidth=1.5
            )
        elif plot_type == "violin":
            sns.violinplot(
                data=df,
                x=condition_col,
                y=metric,
                order=condition_order,
                ax=ax,
                palette="Set2",
                linewidth=1.5
            )
        elif plot_type == "both":
            sns.violinplot(
                data=df,
                x=condition_col,
                y=metric,
                order=condition_order,
                ax=ax,
                palette="Set2",
                linewidth=1.5,
                inner=None,
                alpha=0.6
            )
            sns.boxplot(
                data=df,
                x=condition_col,
                y=metric,
                order=condition_order,
                ax=ax,
                width=0.3,
                palette="Set2",
                linewidth=1.5,
                fliersize=0
            )

        # Customize
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_title(title or f'Score Distribution: {metric.upper()}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_compression_ratio_vs_performance(
        self,
        df: pd.DataFrame,
        performance_metric: str = "f1",
        compression_col: str = "compression_ratio",
        condition_col: str = "condition",
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Scatter plot: compression ratio vs performance.

        Args:
            df: DataFrame with results (must have compression_ratio column)
            performance_metric: Metric for y-axis
            compression_col: Column with compression ratios
            condition_col: Column with condition names
            save_path: Path to save figure
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.default_fig_size)

        # Get unique conditions
        conditions = df[condition_col].unique()

        # Plot each condition
        for condition in conditions:
            condition_df = df[df[condition_col] == condition]

            # Aggregate by document (average across questions)
            if 'doc_id' in condition_df.columns:
                agg_df = condition_df.groupby('doc_id').agg({
                    compression_col: 'first',  # Same for all questions in doc
                    performance_metric: 'mean'
                }).reset_index()
            else:
                agg_df = condition_df

            ax.scatter(
                agg_df[compression_col],
                agg_df[performance_metric],
                label=condition,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )

        # Customize
        ax.set_xlabel('Compression Ratio', fontweight='bold')
        ax.set_ylabel(f'{performance_metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_title(title or 'Compression Ratio vs Performance')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_metric_comparison_heatmap(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ["exact_match", "f1", "semantic_similarity"],
        condition_col: str = "condition",
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Heatmap showing all metrics across conditions.

        Args:
            df: DataFrame with results
            metrics: List of metric columns
            condition_col: Column with condition names
            save_path: Path to save figure
            title: Plot title

        Returns:
            matplotlib Figure
        """
        # Compute mean for each metric and condition
        heatmap_data = df.groupby(condition_col)[metrics].mean()

        # Sort by average across metrics
        heatmap_data['_avg'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('_avg', ascending=False)
        heatmap_data = heatmap_data.drop('_avg', axis=1)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(6, len(heatmap_data) * 0.5)))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            linewidths=1,
            linecolor='gray',
            cbar_kws={'label': 'Score'},
            ax=ax
        )

        # Customize
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Condition', fontweight='bold')
        ax.set_title(title or 'Performance Heatmap Across Metrics')
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_degradation_curve(
        self,
        df: pd.DataFrame,
        baseline_condition: str,
        comparison_conditions: List[str],
        metric: str = "f1",
        condition_col: str = "condition",
        save_path: Optional[Path] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Line plot showing performance degradation from baseline.

        Args:
            df: DataFrame with results
            baseline_condition: Condition to use as baseline (100%)
            comparison_conditions: Conditions to compare
            metric: Metric to plot
            condition_col: Column with condition names
            save_path: Path to save figure
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.default_fig_size)

        # Get baseline performance
        baseline_score = df[df[condition_col] == baseline_condition][metric].mean()

        # Plot each condition
        conditions = [baseline_condition] + comparison_conditions
        positions = list(range(len(conditions)))
        scores = []

        for condition in conditions:
            score = df[df[condition_col] == condition][metric].mean()
            scores.append(score)

        # Calculate degradation percentages
        degradation_pcts = [(baseline_score - s) / baseline_score * 100 if baseline_score > 0 else 0 for s in scores]

        # Create line plot
        ax.plot(positions, scores, marker='o', linewidth=2, markersize=8, label='Absolute Score')
        ax.fill_between(positions, scores, baseline_score, alpha=0.2)

        # Add degradation annotations
        for i, (pos, score, pct) in enumerate(zip(positions, scores, degradation_pcts)):
            if i > 0:  # Skip baseline
                ax.annotate(
                    f'-{pct:.1f}%',
                    xy=(pos, score),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='red'
                )

        # Customize
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_title(title or f'Performance Degradation from Baseline')
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.axhline(y=baseline_score, color='green', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_pairwise_comparison(
        self,
        comparison_results: List,  # List of ComparisonResult objects
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        show_effect_sizes: bool = True
    ) -> plt.Figure:
        """
        Visualization of pairwise statistical comparisons.

        Args:
            comparison_results: List of ComparisonResult objects from StatisticalAnalyzer
            save_path: Path to save figure
            title: Plot title
            show_effect_sizes: Show effect sizes on bars

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, max(6, len(comparison_results) * 0.4)))

        # Prepare data
        labels = []
        mean_diffs = []
        ci_lowers = []
        ci_uppers = []
        colors = []
        effect_sizes = []

        for result in comparison_results:
            label = f"{result.condition_a}\nvs\n{result.condition_b}"
            labels.append(label)
            mean_diffs.append(result.mean_diff)
            ci_lowers.append(result.mean_diff - result.ci_lower)
            ci_uppers.append(result.ci_upper - result.mean_diff)
            effect_sizes.append(result.effect_size)

            # Color by significance
            if result.p_value < 0.001:
                colors.append('darkgreen')
            elif result.p_value < 0.01:
                colors.append('green')
            elif result.p_value < 0.05:
                colors.append('lightgreen')
            else:
                colors.append('gray')

        # Create horizontal bar plot
        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos,
            mean_diffs,
            xerr=[ci_lowers, ci_uppers],
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5,
            capsize=5
        )

        # Add effect size annotations
        if show_effect_sizes:
            for i, (bar, es) in enumerate(zip(bars, effect_sizes)):
                width = bar.get_width()
                ax.text(
                    width + (ci_uppers[i] if width > 0 else -ci_lowers[i]) + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'd={es:.2f}',
                    ha='left' if width > 0 else 'right',
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )

        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Mean Difference (with 95% CI)', fontweight='bold')
        ax.set_title(title or 'Pairwise Statistical Comparisons')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_axisbelow(True)

        # Legend for significance levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='p < 0.001 ***'),
            Patch(facecolor='green', label='p < 0.01 **'),
            Patch(facecolor='lightgreen', label='p < 0.05 *'),
            Patch(facecolor='gray', label='p >= 0.05 ns')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_full_report_figures(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        metrics: List[str] = ["exact_match", "f1", "semantic_similarity"],
        condition_col: str = "condition",
        baseline_condition: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Generate all standard figures for a research report.

        Args:
            df: DataFrame with results
            output_dir: Directory to save figures
            metrics: List of metrics to plot
            condition_col: Column with condition names
            baseline_condition: Baseline for degradation plot

        Returns:
            Dictionary mapping figure names to paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_figures = {}

        print("Generating report figures...")

        # 1. Performance comparison bars for each metric
        for metric in metrics:
            path = output_dir / f"performance_bars_{metric}.png"
            self.plot_performance_comparison(df, metric=metric, save_path=path)
            saved_figures[f"bars_{metric}"] = path

        # 2. Score distributions
        for metric in metrics:
            path = output_dir / f"distribution_{metric}.png"
            self.plot_score_distributions(df, metric=metric, save_path=path, plot_type="both")
            saved_figures[f"dist_{metric}"] = path

        # 3. Metric heatmap
        path = output_dir / "metrics_heatmap.png"
        self.plot_metric_comparison_heatmap(df, metrics=metrics, save_path=path)
        saved_figures["heatmap"] = path

        # 4. Compression ratio vs performance (if available)
        if "compression_ratio" in df.columns:
            for metric in metrics:
                path = output_dir / f"compression_vs_{metric}.png"
                self.plot_compression_ratio_vs_performance(
                    df, performance_metric=metric, save_path=path
                )
                saved_figures[f"compression_{metric}"] = path

        # 5. Degradation curve (if baseline specified)
        if baseline_condition:
            conditions = [c for c in df[condition_col].unique() if c != baseline_condition]
            for metric in metrics:
                path = output_dir / f"degradation_{metric}.png"
                self.plot_degradation_curve(
                    df,
                    baseline_condition=baseline_condition,
                    comparison_conditions=conditions,
                    metric=metric,
                    save_path=path
                )
                saved_figures[f"degradation_{metric}"] = path

        print(f"✓ Generated {len(saved_figures)} figures in {output_dir}")

        return saved_figures
