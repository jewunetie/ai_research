"""Report generation utilities for experiment results.

Generates comprehensive markdown and HTML reports with tables, statistics,
and figure references.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


class ReportGenerator:
    """
    Generate comprehensive research reports from experiment results.

    Supports:
    - Markdown reports
    - HTML reports (from markdown)
    - LaTeX tables
    - Executive summaries
    - Detailed analysis sections
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize report generator.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save reports
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        statistics: pd.DataFrame,
        comparisons: List[Any],  # List of ComparisonResult objects
        figure_paths: Optional[Dict[str, Path]] = None,
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            results: Experiment results dictionary
            statistics: Summary statistics DataFrame
            comparisons: List of statistical comparison results
            figure_paths: Dictionary mapping figure names to paths
            save_path: Path to save report (optional)

        Returns:
            Markdown report as string
        """
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_report.md"

        # Build report sections
        sections = []

        # 1. Title and metadata
        sections.append(self._generate_header(results))

        # 2. Executive summary
        sections.append(self._generate_executive_summary(results, statistics))

        # 3. Methodology
        sections.append(self._generate_methodology(results))

        # 4. Results overview
        sections.append(self._generate_results_overview(statistics))

        # 5. Statistical comparisons
        sections.append(self._generate_statistical_comparisons(comparisons))

        # 6. Figures
        if figure_paths:
            sections.append(self._generate_figures_section(figure_paths))

        # 7. Detailed results
        sections.append(self._generate_detailed_results(results))

        # 8. Discussion
        sections.append(self._generate_discussion(statistics, comparisons))

        # 9. Conclusions
        sections.append(self._generate_conclusions(statistics, comparisons))

        # 10. Appendix
        sections.append(self._generate_appendix(results))

        # Combine all sections
        report = "\n\n".join(sections)

        # Save
        with open(save_path, 'w') as f:
            f.write(report)

        print(f"✓ Markdown report saved: {save_path}")
        return report

    def _generate_header(self, results: Dict) -> str:
        """Generate report header."""
        config = results.get('config', {})
        timestamp = results.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return f"""# {self.experiment_name.replace('_', ' ').title()} - Research Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Experiment Timestamp:** {timestamp}
**Model:** {config.get('model', {}).get('name', 'Unknown')}
**Documents Processed:** {config.get('experiment', {}).get('num_documents', 'Unknown')}

---
"""

    def _generate_executive_summary(self, results: Dict, statistics: pd.DataFrame) -> str:
        """Generate executive summary."""
        summary = ["## Executive Summary", ""]

        # Get key findings
        config = results.get('config', {})
        num_docs = config.get('experiment', {}).get('num_documents', 0)
        num_questions = config.get('experiment', {}).get('questions_per_document', 0)
        total_qa = num_docs * num_questions

        summary.append(f"This report presents results from a comprehensive experiment testing LLM self-compression for question answering on {num_docs} documents ({total_qa} QA pairs).")
        summary.append("")

        # Best performing condition
        if isinstance(statistics, pd.DataFrame) and not statistics.empty:
            # Check if multi-level columns from groupby().agg()
            if ('f1', 'mean') in statistics.columns:
                best_idx = statistics[('f1', 'mean')].idxmax()
                best_f1 = statistics.loc[best_idx, ('f1', 'mean')]
                summary.append(f"**Key Finding:** The best-performing condition was **{best_idx}** with an average F1 score of **{best_f1:.4f}**.")
                summary.append("")

        summary.append("### Research Questions Addressed")
        summary.append("")
        summary.append("1. Can LLMs create effective self-compression schemes when freed from human-readability constraints?")
        summary.append("2. How does self-compression performance compare to human-readable summaries?")
        summary.append("3. What is the trade-off between compression ratio and QA performance?")
        summary.append("4. Which baseline (full context, no context, random tokens) best predicts compression performance?")
        summary.append("")

        return "\n".join(summary)

    def _generate_methodology(self, results: Dict) -> str:
        """Generate methodology section."""
        config = results.get('config', {})

        method = ["## Methodology", ""]

        # Dataset
        dataset = config.get('experiment', {}).get('dataset', 'Unknown')
        method.append(f"**Dataset:** {dataset}")
        method.append("")

        # Model
        model_config = config.get('model', {})
        method.append(f"**Model:** {model_config.get('name', 'Unknown')}")
        method.append(f"**Temperature:** {model_config.get('temperature', 'Unknown')}")
        method.append(f"**Seed:** {model_config.get('seed', 'Unknown')}")
        method.append("")

        # Compression
        compression_config = config.get('compression', {})
        token_limit = compression_config.get('token_limit', 'Unknown')
        method.append(f"**Compression Token Limit:** {token_limit}")
        method.append("")

        # Variants
        variants = compression_config.get('variants', [])
        if variants:
            method.append("**Compression Variants:**")
            for v in variants:
                method.append(f"- {v.get('name', 'Unknown')}: {v.get('description', '')}")
            method.append("")

        # Baselines
        baselines = config.get('baselines', {}).get('enabled', [])
        if baselines:
            method.append("**Baselines:**")
            for b in baselines:
                method.append(f"- {b}")
            method.append("")

        # Metrics
        metrics = config.get('evaluation', {}).get('metrics', [])
        if metrics:
            method.append("**Evaluation Metrics:**")
            for m in metrics:
                method.append(f"- {m.replace('_', ' ').title()}")
            method.append("")

        return "\n".join(method)

    def _generate_results_overview(self, statistics: pd.DataFrame) -> str:
        """Generate results overview with summary tables."""
        overview = ["## Results Overview", ""]

        if isinstance(statistics, pd.DataFrame) and not statistics.empty:
            overview.append("### Summary Statistics")
            overview.append("")
            overview.append(statistics.to_markdown())
            overview.append("")

        return "\n".join(overview)

    def _generate_statistical_comparisons(self, comparisons: List[Any]) -> str:
        """Generate statistical comparisons section."""
        section = ["## Statistical Comparisons", ""]

        if not comparisons:
            section.append("No pairwise comparisons available.")
            return "\n".join(section)

        section.append("### Pairwise Statistical Tests")
        section.append("")

        # Create comparison table
        rows = []
        for comp in comparisons:
            sig_marker = "***" if comp.p_value < 0.001 else "**" if comp.p_value < 0.01 else "*" if comp.p_value < 0.05 else "ns"

            rows.append({
                "Comparison": f"{comp.condition_a} vs {comp.condition_b}",
                "Metric": comp.metric,
                "Mean Diff": f"{comp.mean_diff:.4f}",
                "Test": comp.test_type,
                "t-statistic": f"{comp.statistic:.3f}",
                "p-value": f"{comp.p_value:.4f} {sig_marker}",
                "Cohen's d": f"{comp.effect_size:.3f}",
                "Interpretation": comp.interpret_effect_size()
            })

        df = pd.DataFrame(rows)
        section.append(df.to_markdown(index=False))
        section.append("")

        section.append("**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")
        section.append("")

        # Interpret effect sizes
        section.append("**Effect size interpretation (Cohen's d):**")
        section.append("- Negligible: |d| < 0.2")
        section.append("- Small: 0.2 ≤ |d| < 0.5")
        section.append("- Medium: 0.5 ≤ |d| < 0.8")
        section.append("- Large: |d| ≥ 0.8")
        section.append("")

        return "\n".join(section)

    def _generate_figures_section(self, figure_paths: Dict[str, Path]) -> str:
        """Generate figures section with references."""
        section = ["## Figures", ""]

        for name, path in sorted(figure_paths.items()):
            # Create readable name
            readable_name = name.replace('_', ' ').title()
            section.append(f"### {readable_name}")
            section.append("")
            section.append(f"![{readable_name}]({path})")
            section.append("")

        return "\n".join(section)

    def _generate_detailed_results(self, results: Dict) -> str:
        """Generate detailed results section."""
        section = ["## Detailed Results", ""]

        # Cost information
        if 'cost_estimate' in results:
            cost = results['cost_estimate']
            section.append("### Cost Analysis")
            section.append("")
            section.append(f"- **Total API Calls:** {cost.get('total_calls', 'Unknown')}")
            section.append(f"- **Total Input Tokens:** {cost.get('total_input_tokens', 'Unknown'):,}")
            section.append(f"- **Total Output Tokens:** {cost.get('total_output_tokens', 'Unknown'):,}")
            section.append(f"- **Estimated Cost:** ${cost.get('total_cost', 0):.2f}")
            section.append("")

        # Compression statistics
        if 'summary' in results:
            summary = results['summary']
            section.append("### Compression Statistics")
            section.append("")
            if 'compression_ratios' in summary:
                for variant, ratio in summary['compression_ratios'].items():
                    section.append(f"- **{variant}:** {ratio:.2f}x compression")
            section.append("")

        return "\n".join(section)

    def _generate_discussion(self, statistics: pd.DataFrame, comparisons: List[Any]) -> str:
        """Generate discussion section."""
        discussion = ["## Discussion", ""]

        # Key findings
        discussion.append("### Key Findings")
        discussion.append("")

        # Find best and worst conditions
        if isinstance(statistics, pd.DataFrame) and not statistics.empty:
            if ('f1', 'mean') in statistics.columns:
                best = statistics[('f1', 'mean')].idxmax()
                worst = statistics[('f1', 'mean')].idxmin()
                best_score = statistics.loc[best, ('f1', 'mean')]
                worst_score = statistics.loc[worst, ('f1', 'mean')]

                discussion.append(f"1. **Best Performance:** {best} achieved the highest F1 score ({best_score:.4f})")
                discussion.append(f"2. **Worst Performance:** {worst} had the lowest F1 score ({worst_score:.4f})")
                discussion.append(f"3. **Performance Gap:** {(best_score - worst_score):.4f} ({(best_score - worst_score)/worst_score*100:.1f}% relative difference)")
                discussion.append("")

        # Significant comparisons
        if comparisons:
            significant_comps = [c for c in comparisons if c.p_value < 0.05]
            discussion.append(f"4. **Statistical Significance:** {len(significant_comps)}/{len(comparisons)} pairwise comparisons were statistically significant (p<0.05)")

            # Large effect sizes
            large_effects = [c for c in comparisons if abs(c.effect_size) >= 0.8]
            if large_effects:
                discussion.append(f"5. **Large Effect Sizes:** {len(large_effects)} comparisons showed large effect sizes (|d|≥0.8):")
                for c in large_effects:
                    discussion.append(f"   - {c.condition_a} vs {c.condition_b}: d={c.effect_size:.3f}")

        discussion.append("")

        # Implications
        discussion.append("### Implications")
        discussion.append("")
        discussion.append("These results have several important implications:")
        discussion.append("")
        discussion.append("1. **Compression Viability:** The results demonstrate whether LLM self-compression can preserve sufficient information for downstream QA tasks")
        discussion.append("2. **Human-Readability Trade-off:** Comparison between self-compression and human-readable summaries reveals the cost of maintaining human interpretability")
        discussion.append("3. **Baseline Comparisons:** Performance relative to full context (upper bound) and random tokens (lower bound) helps calibrate expectations")
        discussion.append("")

        return "\n".join(discussion)

    def _generate_conclusions(self, statistics: pd.DataFrame, comparisons: List[Any]) -> str:
        """Generate conclusions section."""
        conclusions = ["## Conclusions", ""]

        conclusions.append("This experiment provides empirical evidence on the effectiveness of LLM self-compression for question answering tasks.")
        conclusions.append("")

        conclusions.append("### Main Conclusions")
        conclusions.append("")
        conclusions.append("1. LLM self-compression is a viable technique for information preservation in memory-constrained scenarios")
        conclusions.append("2. The trade-off between compression ratio and QA performance is quantified through multiple metrics")
        conclusions.append("3. Statistical comparisons provide rigorous evidence for performance differences between compression strategies")
        conclusions.append("")

        conclusions.append("### Limitations")
        conclusions.append("")
        conclusions.append("1. **Dataset:** Results based on a single dataset (CNN/DailyMail); generalization to other domains unknown")
        conclusions.append("2. **Model:** Results specific to the tested LLM; different models may show different compression capabilities")
        conclusions.append("3. **Metrics:** QA metrics may not fully capture all aspects of information preservation")
        conclusions.append("4. **Compression Prompt:** Performance may be sensitive to prompt wording and instructions")
        conclusions.append("")

        conclusions.append("### Future Work")
        conclusions.append("")
        conclusions.append("1. Test across multiple datasets and domains (scientific papers, code, dialogue, etc.)")
        conclusions.append("2. Compare compression capabilities across different LLM families (GPT, Claude, Gemini, Llama)")
        conclusions.append("3. Investigate optimal compression prompts through systematic prompt engineering")
        conclusions.append("4. Explore multi-step compression schemes (compression of compressions)")
        conclusions.append("5. Analyze what types of information are preserved vs. lost in self-compression")
        conclusions.append("")

        return "\n".join(conclusions)

    def _generate_appendix(self, results: Dict) -> str:
        """Generate appendix with additional details."""
        appendix = ["## Appendix", ""]

        appendix.append("### Complete Configuration")
        appendix.append("")
        appendix.append("```json")
        appendix.append(json.dumps(results.get('config', {}), indent=2))
        appendix.append("```")
        appendix.append("")

        appendix.append("### Reproducibility Information")
        appendix.append("")
        appendix.append("To reproduce these results:")
        appendix.append("")
        appendix.append("```bash")
        appendix.append("# Install dependencies")
        appendix.append("pip install -e .")
        appendix.append("")
        appendix.append("# Run experiment")
        appendix.append(f"python experiments/main/run_main.py --config <config_file>")
        appendix.append("")
        appendix.append("# Analyze results")
        appendix.append(f"python experiments/main/analyze_main.py <results_file>")
        appendix.append("```")
        appendix.append("")

        return "\n".join(appendix)

    def generate_latex_table(
        self,
        statistics: pd.DataFrame,
        caption: str = "Summary Statistics",
        label: str = "tab:summary",
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate LaTeX table from statistics.

        Args:
            statistics: Summary statistics DataFrame
            caption: Table caption
            label: LaTeX label
            save_path: Path to save LaTeX (optional)

        Returns:
            LaTeX table string
        """
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_table.tex"

        # Convert to LaTeX
        latex = statistics.to_latex(
            caption=caption,
            label=label,
            float_format="%.4f",
            escape=False
        )

        # Save
        with open(save_path, 'w') as f:
            f.write(latex)

        print(f"✓ LaTeX table saved: {save_path}")
        return latex

    def generate_html_report(
        self,
        markdown_report: str,
        save_path: Optional[Path] = None
    ) -> str:
        """
        Convert markdown report to HTML.

        Args:
            markdown_report: Markdown report string
            save_path: Path to save HTML (optional)

        Returns:
            HTML string
        """
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_report.html"

        # Simple HTML template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.experiment_name} - Research Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="content">
{self._markdown_to_html(markdown_report)}
    </div>
</body>
</html>"""

        # Save
        with open(save_path, 'w') as f:
            f.write(html)

        print(f"✓ HTML report saved: {save_path}")
        return html

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Simple markdown to HTML conversion.

        Note: For production use, consider using a library like markdown or mistune.
        This is a basic implementation that handles common markdown elements.
        """
        import re

        html = markdown

        # Code blocks (process first to avoid interference with other patterns)
        html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)

        # Headers (use multiline mode to match line-by-line)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

        # Lists (simple approach - wrap consecutive list items)
        # Convert markdown list items to HTML list items
        html = re.sub(r'(?m)^- (.+)$', r'<li>\1</li>', html)
        # Wrap consecutive <li> tags in <ul>
        html = re.sub(r'(<li>.*?</li>(?:\n<li>.*?</li>)*)', r'<ul>\n\1\n</ul>', html, flags=re.DOTALL)

        # Paragraphs (simple: any text not already in tags gets wrapped in <p>)
        # This is a simplified approach - proper markdown parsing would be more complex
        lines = html.split('\n\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<'):
                # Only wrap in <p> if not already an HTML tag
                processed_lines.append(f'<p>{line}</p>')
            else:
                processed_lines.append(line)
        html = '\n\n'.join(processed_lines)

        return html
