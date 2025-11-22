"""
Baseline Chain-of-Thought prompting.

Simple, standard prompting without MECE guidance.
"""

from typing import Dict, Any


def create_baseline_prompt(problem: Dict[str, Any]) -> str:
    """
    Create baseline CoT prompt for a problem.

    Args:
        problem: Problem dictionary with 'problem' field

    Returns:
        Formatted prompt string
    """
    problem_text = problem["problem"]

    prompt = f"""Problem: {problem_text}

Let's solve this step by step:"""

    return prompt


def create_baseline_prompt_with_thinking(problem: Dict[str, Any]) -> str:
    """
    Create baseline CoT prompt with explicit thinking mode instruction.

    Args:
        problem: Problem dictionary with 'problem' field

    Returns:
        Formatted prompt string with thinking mode enabled
    """
    problem_text = problem["problem"]

    prompt = f"""You are a helpful AI that thinks step by step. Show your reasoning process clearly before giving the final answer.

Problem: {problem_text}

Let's solve this step by step:"""

    return prompt


# Alias for the recommended version
create_prompt = create_baseline_prompt_with_thinking
