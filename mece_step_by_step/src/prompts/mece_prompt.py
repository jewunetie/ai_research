"""
MECE (Mutually Exclusive, Collectively Exhaustive) prompting.

Multiple versions of varying structure to test with small model.
"""

from typing import Dict, Any


def create_mece_prompt_v1(problem: Dict[str, Any]) -> str:
    """
    MECE Prompt Version 1: Explicit MECE instruction.

    Most natural, explains MECE principle directly.

    Args:
        problem: Problem dictionary with 'problem' field

    Returns:
        Formatted prompt string
    """
    problem_text = problem["problem"]

    prompt = f"""You are a helpful AI that thinks step by step. Show your reasoning process clearly before giving the final answer.

Problem: {problem_text}

Solve this problem using the MECE principle (Mutually Exclusive, Collectively Exhaustive):

1. Identify all distinct cases that need to be considered
2. Ensure cases don't overlap (mutually exclusive)
3. Ensure all possibilities are covered (collectively exhaustive)
4. Solve each case separately
5. Combine results

Let's solve this step by step:"""

    return prompt


def create_mece_prompt_v2(problem: Dict[str, Any]) -> str:
    """
    MECE Prompt Version 2: Case analysis template.

    More structured, breaks down the MECE process.

    Args:
        problem: Problem dictionary with 'problem' field

    Returns:
        Formatted prompt string
    """
    problem_text = problem["problem"]

    prompt = f"""You are a helpful AI that thinks step by step. Show your reasoning process clearly before giving the final answer.

Problem: {problem_text}

Analyze this problem by breaking it into cases:

Step 1: Identify what cases need to be considered
Step 2: For each case, state the condition clearly
Step 3: Verify cases don't overlap
Step 4: Verify all possibilities are covered
Step 5: Solve each case
Step 6: Combine solutions

Let's work through this:"""

    return prompt


def create_mece_prompt_v3(problem: Dict[str, Any]) -> str:
    """
    MECE Prompt Version 3: Highly structured template.

    Most structured, provides explicit sections. Use if v1 and v2 underperform.

    Args:
        problem: Problem dictionary with 'problem' field

    Returns:
        Formatted prompt string
    """
    problem_text = problem["problem"]

    prompt = f"""You are a helpful AI that thinks step by step. Show your reasoning process clearly before giving the final answer.

Problem: {problem_text}

Use structured case analysis:

CASE IDENTIFICATION:
- List all cases needed: [...]

MUTUAL EXCLUSIVITY CHECK:
- Verify no cases overlap: [...]

EXHAUSTIVENESS CHECK:
- Verify all possibilities covered: [...]

CASE SOLUTIONS:
- Case 1: [condition] → [solution]
- Case 2: [condition] → [solution]
...

FINAL ANSWER:
[Combined answer]

Let's solve:"""

    return prompt


# Default to Version 1 (most natural)
create_prompt = create_mece_prompt_v1


# Version selector
def create_mece_prompt(problem: Dict[str, Any], version: int = 1) -> str:
    """
    Create MECE prompt with specified version.

    Args:
        problem: Problem dictionary
        version: Prompt version (1, 2, or 3)

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If version is invalid
    """
    if version == 1:
        return create_mece_prompt_v1(problem)
    elif version == 2:
        return create_mece_prompt_v2(problem)
    elif version == 3:
        return create_mece_prompt_v3(problem)
    else:
        raise ValueError(f"Invalid MECE prompt version: {version}. Must be 1, 2, or 3.")
