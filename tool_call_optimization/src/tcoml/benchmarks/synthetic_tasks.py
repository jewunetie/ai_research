import random
from typing import List
from .task_models import Task, TaskDifficulty


class SyntheticTaskGenerator:
    """Generate synthetic tasks for testing and training"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_calculator_tasks(self, n: int = 20) -> List[Task]:
        """
        Generate calculator tasks.

        Examples:
        - "Calculate 15% tip on $48.50"
        - "What is 234 * 67?"
        - "Find the average of 10, 20, 30, 40"
        """
        tasks = []

        # Tip calculations
        for i in range(n // 4):
            amount = round(random.uniform(10, 200), 2)
            tip_pct = random.choice([10, 15, 18, 20, 25])
            expected = round(amount * tip_pct / 100, 2)

            tasks.append(Task(
                id=f"calc_tip_{i:03d}",
                description=f"Calculate {tip_pct}% tip on ${amount}",
                required_tools=["calculator"],
                success_criteria=lambda output, exp=expected: abs(float(output) - exp) < 0.01,
                difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
                category="calculator",
                ground_truth=expected
            ))

        # Multiplication
        for i in range(n // 4):
            a = random.randint(100, 999)
            b = random.randint(10, 99)
            expected = a * b

            tasks.append(Task(
                id=f"calc_mult_{i:03d}",
                description=f"What is {a} * {b}?",
                required_tools=["calculator"],
                success_criteria=lambda output, exp=expected: int(output) == exp,
                difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
                category="calculator",
                ground_truth=expected
            ))

        # Average calculations
        for i in range(n // 4):
            numbers = [random.randint(1, 100) for _ in range(random.randint(3, 6))]
            expected = round(sum(numbers) / len(numbers), 2)

            tasks.append(Task(
                id=f"calc_avg_{i:03d}",
                description=f"Find the average of {', '.join(map(str, numbers))}",
                required_tools=["calculator"],
                success_criteria=lambda output, exp=expected: abs(float(output) - exp) < 0.01,
                difficulty=TaskDifficulty.LEVEL_2_MULTI_TOOL,
                category="calculator",
                ground_truth=expected,
                metadata={"numbers": numbers}
            ))

        # Complex calculations (requires multiple steps)
        for i in range(n // 4):
            base = random.randint(100, 1000)
            rate = random.randint(3, 10)
            years = random.randint(1, 5)
            # Simple interest: P * r * t / 100
            expected = round(base * rate * years / 100, 2)

            tasks.append(Task(
                id=f"calc_interest_{i:03d}",
                description=f"Calculate simple interest: ${base} principal, {rate}% annual rate, {years} years",
                required_tools=["calculator"],
                success_criteria=lambda output, exp=expected: abs(float(output) - exp) < 0.01,
                difficulty=TaskDifficulty.LEVEL_2_MULTI_TOOL,
                category="calculator",
                ground_truth=expected
            ))

        return tasks[:n]

    def generate_string_tasks(self, n: int = 20) -> List[Task]:
        """
        Generate string manipulation tasks.

        Examples:
        - "Count the number of words in 'Hello world'"
        - "Reverse the string 'python'"
        - "Convert 'test' to uppercase"
        """
        tasks = []

        # Word count
        for i in range(n // 3):
            sentence = " ".join(random.choices(
                ["hello", "world", "python", "programming", "test", "code"],
                k=random.randint(3, 8)
            ))
            expected = len(sentence.split())

            tasks.append(Task(
                id=f"str_count_{i:03d}",
                description=f"Count the number of words in '{sentence}'",
                required_tools=["string_processor"],
                success_criteria=lambda output, exp=expected: int(output) == exp,
                difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
                category="string",
                ground_truth=expected
            ))

        # String reversal
        for i in range(n // 3):
            word = random.choice(["python", "javascript", "programming", "algorithm"])
            expected = word[::-1]

            tasks.append(Task(
                id=f"str_reverse_{i:03d}",
                description=f"Reverse the string '{word}'",
                required_tools=["string_processor"],
                success_criteria=lambda output, exp=expected: output.strip() == exp,
                difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
                category="string",
                ground_truth=expected
            ))

        # Case conversion
        for i in range(n // 3):
            word = random.choice(["test", "example", "demo", "sample"])
            op = random.choice(["uppercase", "lowercase"])
            expected = word.upper() if op == "uppercase" else word.lower()

            tasks.append(Task(
                id=f"str_case_{i:03d}",
                description=f"Convert '{word}' to {op}",
                required_tools=["string_processor"],
                success_criteria=lambda output, exp=expected: output.strip() == exp,
                difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
                category="string",
                ground_truth=expected
            ))

        return tasks[:n]

    def generate_multi_step_tasks(self, n: int = 20) -> List[Task]:
        """
        Generate tasks requiring multiple tools in sequence.

        Examples:
        - "Calculate 10 + 20, then convert the result to a string and reverse it"
        - "Count words in 'hello world', multiply by 5"
        """
        tasks = []

        for i in range(n):
            # Calc then string
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            sum_ab = a + b
            expected = str(sum_ab)[::-1]

            tasks.append(Task(
                id=f"multi_calc_str_{i:03d}",
                description=f"Calculate {a} + {b}, then reverse the result as a string",
                required_tools=["calculator", "string_processor"],
                success_criteria=lambda output, exp=expected: str(output).strip() == exp,
                difficulty=TaskDifficulty.LEVEL_2_MULTI_TOOL,
                category="multi_step",
                ground_truth=expected
            ))

        return tasks

    def generate_all(self, n_per_type: int = 10) -> List[Task]:
        """Generate all task types"""
        tasks = []
        tasks.extend(self.generate_calculator_tasks(n_per_type))
        tasks.extend(self.generate_string_tasks(n_per_type))
        tasks.extend(self.generate_multi_step_tasks(n_per_type))
        return tasks
