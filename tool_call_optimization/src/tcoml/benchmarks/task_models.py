from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TaskDifficulty(int, Enum):
    """Task difficulty levels for curriculum learning"""
    LEVEL_1_SINGLE_TOOL = 1
    LEVEL_2_MULTI_TOOL = 2
    LEVEL_3_CONDITIONAL = 3
    LEVEL_4_ERROR_RECOVERY = 4
    LEVEL_5_COMPLEX = 5


class Task(BaseModel):
    """
    Represents a task that the agent must complete using tools.

    Example:
        task = Task(
            id="calc_001",
            description="Calculate 15% tip on $48.50",
            required_tools=["calculator"],
            success_criteria=lambda output: abs(float(output) - 7.275) < 0.01,
            difficulty=TaskDifficulty.LEVEL_1_SINGLE_TOOL,
            category="calculator",
            ground_truth=7.275
        )
    """
    id: str
    description: str
    required_tools: List[str]
    success_criteria: Callable[[Any], bool] = Field(exclude=True)
    difficulty: TaskDifficulty = TaskDifficulty.LEVEL_1_SINGLE_TOOL
    category: str = "general"
    ground_truth: Optional[Any] = None
    test_function: Optional[Callable[[Any], Dict[str, Any]]] = Field(default=None, exclude=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def evaluate(self, output: Any) -> bool:
        """Evaluate if the output meets success criteria"""
        try:
            return self.success_criteria(output)
        except Exception as e:
            return False

    def run_tests(self, output: Any) -> Dict[str, Any]:
        """Run test function if available"""
        if self.test_function:
            return self.test_function(output)
        return {"passed": self.evaluate(output), "details": "Basic evaluation"}
