"""
Parsing utilities for extracting reasoning steps, cases, and solutions from model outputs.

These parsers are designed to handle free-form text from LLMs and extract
structured information for MECE metric computation.
"""

import re
from typing import List, Set, Optional


def parse_reasoning_steps(response: str) -> List[str]:
    """
    Extract individual reasoning steps from model response.

    Heuristics:
    - Look for numbered steps (1., 2., Step 1:, etc.)
    - Look for bullet points (-, *, •)
    - Look for "Case" keywords
    - Fall back to sentence splitting

    Args:
        response: Model's full response text

    Returns:
        List of reasoning step strings

    Example:
        >>> response = "Step 1: Check x >= 0\\nStep 2: Check x < 0"
        >>> parse_reasoning_steps(response)
        ['Check x >= 0', 'Check x < 0']
    """
    if not response or not response.strip():
        return []

    steps = []

    # Pattern 1: Numbered steps (1., 2., etc.)
    numbered_pattern = r'(?:^|\n)\s*(\d+\.)\s+(.+?)(?=\n\s*\d+\.|$)'
    numbered_matches = re.findall(numbered_pattern, response, re.MULTILINE | re.DOTALL)
    if numbered_matches:
        steps.extend([match[1].strip() for match in numbered_matches])

    # Pattern 2: "Step N:" format
    step_pattern = r'(?:^|\n)\s*Step\s+\d+:\s*(.+?)(?=\n\s*Step\s+\d+:|$)'
    step_matches = re.findall(step_pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if step_matches:
        steps.extend([match.strip() for match in step_matches])

    # Pattern 3: "Case N:" format
    case_pattern = r'(?:^|\n)\s*Case\s+\d+:\s*(.+?)(?=\n\s*Case\s+\d+:|$)'
    case_matches = re.findall(case_pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if case_matches:
        steps.extend([match.strip() for match in case_matches])

    # Pattern 4: Bullet points (-, *, •)
    bullet_pattern = r'(?:^|\n)\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|$)'
    bullet_matches = re.findall(bullet_pattern, response, re.MULTILINE | re.DOTALL)
    if bullet_matches:
        steps.extend([match.strip() for match in bullet_matches])

    # If we found steps, return them (deduplicated)
    if steps:
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in steps:
            step_lower = step.lower()
            if step_lower not in seen and len(step) > 10:  # Filter out very short steps
                seen.add(step_lower)
                unique_steps.append(step)
        return unique_steps

    # Fallback: Split by sentences
    sentences = re.split(r'[.!?]\s+', response)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_case_conditions(reasoning_steps: List[str]) -> List[str]:
    """
    Extract case conditions from reasoning steps.

    Looks for mathematical conditions like:
    - "when x >= 0"
    - "if x < 3"
    - "for x > 5"
    - "x - 3 >= 0"

    Args:
        reasoning_steps: List of reasoning step strings

    Returns:
        List of extracted condition strings

    Example:
        >>> steps = ["When x >= 0, we have x = 2x - 3", "When x < 0, we have -x = 2x - 3"]
        >>> extract_case_conditions(steps)
        ['x >= 0', 'x < 0']
    """
    conditions = []

    # Pattern for "when/if/for X [condition]"
    condition_keywords = r'(?:when|if|for|where|given|assuming)\s+'

    # Math condition patterns
    patterns = [
        # "when x >= 0"
        condition_keywords + r'([a-zA-Z0-9\s\-+*/\(\)]+(?:>=|<=|>|<|=)\s*[a-zA-Z0-9\s\-+*/\(\)]+)',
        # "x >= 0 (when...)"
        r'([a-zA-Z0-9\s\-+*/\(\)]+(?:>=|<=|>|<|=)\s*[a-zA-Z0-9\s\-+*/\(\)]+)\s*\(',
        # Standalone: "x >= 0"
        r'\b([a-zA-Z]\s*(?:>=|<=|>|<)\s*[0-9\-]+)\b',
    ]

    for step in reasoning_steps:
        for pattern in patterns:
            matches = re.findall(pattern, step, re.IGNORECASE)
            for match in matches:
                condition = match.strip()
                # Clean up condition
                condition = re.sub(r'\s+', ' ', condition)  # Normalize whitespace
                if condition and len(condition) > 1:
                    conditions.append(condition)

    return conditions


def count_overlapping_conditions(conditions: List[str]) -> int:
    """
    Count pairs of conditions that overlap (violate mutual exclusivity).

    This is a heuristic check looking for obvious overlaps like:
    - "x >= 0" and "x > 0" (overlap)
    - "x < 3" and "x <= 3" (overlap)

    Args:
        conditions: List of condition strings

    Returns:
        Number of overlapping condition pairs

    Note:
        This is a simple heuristic. Perfect overlap detection would require
        symbolic math, which is beyond scope.

    Example:
        >>> conditions = ["x >= 0", "x > 0"]
        >>> count_overlapping_conditions(conditions)
        1  # These overlap
    """
    if len(conditions) < 2:
        return 0

    overlaps = 0

    # Check each pair
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            cond1 = conditions[i]
            cond2 = conditions[j]

            # Heuristic: Check if one condition is strictly contained in another
            # Example: "x >= 0" contains "x > 0"
            if _conditions_overlap(cond1, cond2):
                overlaps += 1

    return overlaps


def _conditions_overlap(cond1: str, cond2: str) -> bool:
    """
    Heuristic check if two conditions overlap.

    Args:
        cond1: First condition
        cond2: Second condition

    Returns:
        True if conditions likely overlap
    """
    # Extract variable and comparison
    pattern = r'([a-zA-Z]+)\s*(>=|<=|>|<|=)\s*([0-9\-]+)'

    match1 = re.search(pattern, cond1)
    match2 = re.search(pattern, cond2)

    if not match1 or not match2:
        return False

    var1, op1, val1 = match1.groups()
    var2, op2, val2 = match2.groups()

    # Must be same variable
    if var1 != var2:
        return False

    # Same value with different operators = overlap
    # e.g., "x >= 0" and "x > 0"
    if val1 == val2:
        overlapping_ops = [
            ('>=', '>'), ('>', '>='),
            ('<=', '<'), ('<', '<='),
            ('>=', '='), ('=', '>='),
            ('<=', '='), ('=', '<='),
        ]
        if (op1, op2) in overlapping_ops:
            return True

    return False


def parse_solutions(text: str) -> Set[str]:
    """
    Extract solutions from text.

    Looks for patterns like:
    - "x = 3"
    - "x = -2"
    - "x ∈ {1, 2, 3}"
    - "x = 3 or x = -2"

    Args:
        text: Text containing solutions

    Returns:
        Set of solution strings (normalized)

    Example:
        >>> parse_solutions("The solution is x = 3 and x = -2")
        {'x = 3', 'x = -2'}
    """
    solutions = set()

    # Pattern 1: "x = <number>"
    pattern1 = r'x\s*=\s*([0-9\-\.√]+)'
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    for match in matches1:
        solutions.add(f"x = {match.strip()}")

    # Pattern 2: "x ∈ {...}"
    pattern2 = r'x\s*∈\s*\{([^}]+)\}'
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        # Split by commas
        values = [v.strip() for v in match.split(',')]
        for val in values:
            solutions.add(f"x ∈ {{{val}}}" if '∈' in text else f"x = {val}")

    # Pattern 3: Interval notation like "-2 <= x <= 3" or "x < -3 or x > 2"
    pattern3 = r'([0-9\-]+\s*[<>=]+\s*x\s*[<>=]+\s*[0-9\-]+|x\s*[<>=]+\s*[0-9\-]+)'
    matches3 = re.findall(pattern3, text)
    for match in matches3:
        solutions.add(match.strip())

    return solutions


def normalize_solution(solution: str) -> str:
    """
    Normalize a solution string for comparison.

    Args:
        solution: Solution string

    Returns:
        Normalized solution string (lowercase)

    Example:
        >>> normalize_solution("x=3")
        'x = 3'
        >>> normalize_solution("X = 3")
        'x = 3'
        >>> normalize_solution("  x  =  -2  ")
        'x = -2'
    """
    # Remove extra whitespace
    solution = re.sub(r'\s+', ' ', solution.strip())

    # Normalize "x=" to "x ="
    solution = re.sub(r'([a-zA-Z])=', r'\1 = ', solution)
    solution = re.sub(r'=([0-9])', r'= \1', solution)

    # Convert to lowercase for case-insensitive matching
    solution = solution.lower()

    return solution.strip()
