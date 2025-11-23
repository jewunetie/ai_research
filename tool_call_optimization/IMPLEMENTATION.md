# Implementation Guide: Tool Call Optimization via Meta-Learning

## Document Overview

This document provides detailed implementation guidance for building the Tool Call Optimization system. It translates the research findings (RESEARCH.md) and project specification (CLAUDE.md) into actionable development steps.

**Status**: Implementation Blueprint
**Last Updated**: 2025-11-22
**Prerequisite Reading**: CLAUDE.md, RESEARCH.md (Sections 1-11)

---

## Table of Contents

1. [Pre-Implementation Setup](#1-pre-implementation-setup)
2. [Project Structure](#2-project-structure)
3. [Development Environment](#3-development-environment)
4. [Phase 1: Foundation](#4-phase-1-foundation)
5. [Phase 2: Feedback Loop](#5-phase-2-feedback-loop)
6. [Phase 3: Meta-Prompting](#6-phase-3-meta-prompting)
7. [Phase 4: Evaluation](#7-phase-4-evaluation)
8. [Phase 5: Real Benchmarks](#8-phase-5-real-benchmarks)
9. [Phase 6: Analysis & Documentation](#9-phase-6-analysis--documentation)
10. [Testing Strategy](#10-testing-strategy)
11. [Configuration Management](#11-configuration-management)
12. [Common Pitfalls & Solutions](#12-common-pitfalls--solutions)

---

## 1. Pre-Implementation Setup

### 1.1 Environment Verification

**Before writing any code, verify current state:**

```bash
# 1. Check Python version
python --version  # Should be 3.10+, may need 3.12+

# 2. Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Verify Git
git --version

# 4. Check available compute
nvidia-smi  # If using GPU
```

### 1.2 API Setup

**Choose your LLM provider and set up access:**

**Option A: OpenAI**
```bash
# Sign up at platform.openai.com
# Get API key
export OPENAI_API_KEY="your-key-here"

# Test current API
python -c "import openai; print(openai.__version__)"

# Verify current models available
# Check: platform.openai.com/docs/models
```

**Option B: Anthropic**
```bash
# Sign up at console.anthropic.com
# Get API key
export ANTHROPIC_API_KEY="your-key-here"

# Test current API
python -c "import anthropic; print(anthropic.__version__)"
```

**Option C: Local Models**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (verify latest available)
ollama pull llama3.2:3b  # Or current best 3B model

# Test
ollama run llama3.2:3b "Hello"
```

### 1.3 Research Current State

**Critical: Don't assume outdated information**

Create a research document `docs/current_state.md`:

```markdown
# Current State Research (as of [DATE])

## Models Tested
- [ ] OpenAI: Latest model is _____, API version _____
- [ ] Anthropic: Latest model is _____, API version _____
- [ ] Local: Best 3B model is _____, best 7B is _____

## API Capabilities
- [ ] Tool/function calling: Supported in _____
- [ ] Structured outputs: Available via _____
- [ ] Batch processing: _____
- [ ] Cost per 1M tokens: Input $_____, Output $_____

## Framework Versions
- [ ] DSPy: Current version _____
- [ ] LangChain: Current version _____
- [ ] Pydantic: Current version _____

## Benchmarks
- [ ] SWE-Bench: Current version _____
- [ ] ToolBench: Current version _____
```

Fill this out BEFORE starting implementation.

---

## 2. Project Structure

### 2.1 Directory Layout

Create the following structure:

```
tool_call_optimization/
├── src/
│   ├── tcoml/                    # Main package
│   │   ├── __init__.py
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── tool_call_agent.py
│   │   │   ├── tools.py
│   │   │   └── tool_registry.py
│   │   ├── tracing/
│   │   │   ├── __init__.py
│   │   │   ├── execution_tracer.py
│   │   │   └── trace_models.py
│   │   ├── feedback/
│   │   │   ├── __init__.py
│   │   │   ├── feedback_generator.py
│   │   │   └── feedback_aggregator.py
│   │   ├── optimization/
│   │   │   ├── __init__.py
│   │   │   ├── meta_prompter.py
│   │   │   └── optimization_loop.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   └── evaluator.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── synthetic_tasks.py
│   │   │   ├── task_models.py
│   │   │   └── generators.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base_client.py
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── local_client.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       └── config.py
│   └── scripts/
│       ├── run_optimization.py
│       ├── evaluate_baseline.py
│       └── analyze_results.py
├── tests/
│   ├── unit/
│   │   ├── test_agent.py
│   │   ├── test_tracer.py
│   │   ├── test_feedback.py
│   │   └── test_optimization.py
│   ├── integration/
│   │   └── test_full_loop.py
│   └── fixtures/
│       └── sample_tasks.py
├── notebooks/
│   ├── 01_explore_api.ipynb
│   ├── 02_test_feedback.ipynb
│   └── 03_visualize_results.ipynb
├── data/
│   ├── traces/
│   ├── prompts/
│   ├── results/
│   └── benchmarks/
├── configs/
│   ├── default.yaml
│   ├── openai.yaml
│   ├── anthropic.yaml
│   └── local.yaml
├── docs/
│   ├── current_state.md       # Your research
│   ├── api_examples.md
│   └── troubleshooting.md
├── pyproject.toml
├── uv.lock
├── README.md
├── CLAUDE.md                   # Project overview
├── RESEARCH.md                 # Research foundations
├── IMPLEMENTATION.md           # This file
└── .gitignore
```

### 2.2 Create Initial Structure

```bash
cd tool_call_optimization

# Create directories
mkdir -p src/tcoml/{agent,tracing,feedback,optimization,evaluation,benchmarks,llm,utils}
mkdir -p src/scripts
mkdir -p tests/{unit,integration,fixtures}
mkdir -p notebooks
mkdir -p data/{traces,prompts,results,benchmarks}
mkdir -p configs
mkdir -p docs

# Create __init__.py files
touch src/tcoml/__init__.py
touch src/tcoml/{agent,tracing,feedback,optimization,evaluation,benchmarks,llm,utils}/__init__.py

# Create placeholder files (we'll implement these)
touch src/tcoml/agent/{tool_call_agent,tools,tool_registry}.py
touch src/tcoml/tracing/{execution_tracer,trace_models}.py
touch src/tcoml/feedback/{feedback_generator,feedback_aggregator}.py
touch src/tcoml/optimization/{meta_prompter,optimization_loop}.py
touch src/tcoml/evaluation/{metrics,evaluator}.py
touch src/tcoml/benchmarks/{synthetic_tasks,task_models,generators}.py
touch src/tcoml/llm/{base_client,openai_client,anthropic_client,local_client}.py
touch src/tcoml/utils/{logging,config}.py
```

---

## 3. Development Environment

### 3.1 Initialize Project

**Create `pyproject.toml`:**

```toml
[project]
name = "tcoml"
version = "0.1.0"
description = "Tool Call Optimization via Meta-Learning"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0",
    "anthropic>=0.7",
    "pydantic>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "tiktoken>=0.5",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "ipython>=8.0",
    "jupyter>=1.0",
]
local = [
    "transformers>=4.35",
    "torch>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 3.2 Install Dependencies

```bash
# Initialize uv environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import tcoml; import openai; import pydantic; print('All imports successful')"
```

### 3.3 Environment Variables

**Create `.env` file:**

```bash
# API Keys (NEVER commit this file!)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Configuration
TCOML_MODEL_PROVIDER=openai  # or anthropic, local
TCOML_LOG_LEVEL=INFO
TCOML_DATA_DIR=./data
TCOML_CACHE_DIR=./data/.cache

# Experiment Settings
TCOML_RANDOM_SEED=42
TCOML_MAX_ITERATIONS=10
TCOML_CONVERGENCE_THRESHOLD=0.02
```

**Create `.gitignore`:**

```
# Environment
.env
.venv/
venv/

# Data
data/traces/*
data/results/*
!data/traces/.gitkeep
!data/results/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Model checkpoints
*.ckpt
*.pth
```

---

## 4. Phase 1: Foundation (Weeks 1-2)

### 4.1 Task Models (Day 1-2)

**File: `src/tcoml/benchmarks/task_models.py`**

```python
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
```

### 4.2 Tool Definition (Day 2-3)

**File: `src/tcoml/agent/tools.py`**

```python
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Parameter definition for a tool"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None


class Tool(BaseModel):
    """
    Tool that can be called by the agent.

    Example:
        def add(a: float, b: float) -> float:
            return a + b

        calculator = Tool(
            name="calculator_add",
            description="Add two numbers",
            parameters=[
                ToolParameter(name="a", type="number", description="First number"),
                ToolParameter(name="b", type="number", description="Second number"),
            ],
            returns="Sum of a and b",
            function=add
        )
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str
    function: Callable = Field(exclude=True)
    error_messages: Dict[str, str] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with given arguments.

        Returns:
            Dict with 'success', 'result', 'error' keys
        """
        try:
            result = self.function(**kwargs)
            return {
                "success": True,
                "result": result,
                "tool": self.name,
                "args": kwargs
            }
        except Exception as e:
            error_type = type(e).__name__
            error_msg = self.error_messages.get(error_type, str(e))
            return {
                "success": False,
                "error": error_msg,
                "error_type": error_type,
                "tool": self.name,
                "args": kwargs
            }

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **({"enum": param.enum} if param.enum else {})
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }
```

### 4.3 Synthetic Task Generator (Day 3-5)

**File: `src/tcoml/benchmarks/synthetic_tasks.py`**

```python
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
```

### 4.4 Basic Tools Implementation (Day 5-7)

**File: `src/tcoml/agent/tool_registry.py`**

```python
from typing import Dict, List
from .tools import Tool, ToolParameter


class ToolRegistry:
    """Registry of available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name]

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def _register_default_tools(self) -> None:
        """Register default synthetic tools"""

        # Calculator tool
        def calculator(operation: str, a: float, b: float = None) -> float:
            """Perform basic calculator operations"""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                if b == 0:
                    raise ValueError("Division by zero")
                return a / b
            elif operation == "percentage":
                # a is base, b is percentage
                return a * b / 100
            else:
                raise ValueError(f"Unknown operation: {operation}")

        self.register(Tool(
            name="calculator",
            description="Perform basic arithmetic operations",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    enum=["add", "subtract", "multiply", "divide", "percentage"]
                ),
                ToolParameter(name="a", type="number", description="First number"),
                ToolParameter(name="b", type="number", description="Second number", required=False),
            ],
            returns="Result of the calculation",
            function=calculator,
            error_messages={
                "ValueError": "Invalid operation or division by zero"
            }
        ))

        # String processor tool
        def string_processor(operation: str, text: str) -> str:
            """Perform string operations"""
            if operation == "count_words":
                return str(len(text.split()))
            elif operation == "reverse":
                return text[::-1]
            elif operation == "uppercase":
                return text.upper()
            elif operation == "lowercase":
                return text.lower()
            elif operation == "length":
                return str(len(text))
            else:
                raise ValueError(f"Unknown operation: {operation}")

        self.register(Tool(
            name="string_processor",
            description="Perform string manipulation operations",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    enum=["count_words", "reverse", "uppercase", "lowercase", "length"]
                ),
                ToolParameter(name="text", type="string", description="Input text"),
            ],
            returns="Result of the string operation",
            function=string_processor
        ))
```

**Checkpoint**: At this point you should be able to:
```python
from tcoml.benchmarks.task_models import Task, TaskDifficulty
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator
from tcoml.agent.tool_registry import ToolRegistry

# Generate tasks
generator = SyntheticTaskGenerator()
tasks = generator.generate_calculator_tasks(5)
print(f"Generated {len(tasks)} tasks")

# Test tools
registry = ToolRegistry()
result = registry.get("calculator").execute(operation="add", a=10, b=20)
print(f"Calculator result: {result}")
```

### 4.5 Trace Models (Day 7-9)

**File: `src/tcoml/tracing/trace_models.py`**

```python
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ToolCall(BaseModel):
    """Record of a single tool call"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float = 0.0


class ExecutionTrace(BaseModel):
    """Complete execution trace for a task attempt"""
    task_id: str
    task_description: str
    system_prompt: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_output: Any = None
    task_success: bool = False
    total_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to the trace"""
        self.tool_calls.append(tool_call)

    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Summarize tool calls for analysis"""
        return {
            "total_calls": len(self.tool_calls),
            "successful_calls": sum(1 for tc in self.tool_calls if tc.success),
            "failed_calls": sum(1 for tc in self.tool_calls if not tc.success),
            "unique_tools": len(set(tc.tool_name for tc in self.tool_calls)),
            "total_duration_ms": sum(tc.duration_ms for tc in self.tool_calls),
        }


class TraceDataset(BaseModel):
    """Collection of execution traces"""
    traces: List[ExecutionTrace] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Add a trace to the dataset"""
        self.traces.append(trace)

    def filter_by_success(self, success: bool) -> "TraceDataset":
        """Filter traces by success status"""
        filtered = [t for t in self.traces if t.task_success == success]
        return TraceDataset(traces=filtered, metadata=self.metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.traces:
            return {"error": "No traces in dataset"}

        return {
            "total_traces": len(self.traces),
            "successful_traces": sum(1 for t in self.traces if t.task_success),
            "success_rate": sum(1 for t in self.traces if t.task_success) / len(self.traces),
            "avg_tool_calls": sum(len(t.tool_calls) for t in self.traces) / len(self.traces),
            "avg_duration_ms": sum(t.total_time_ms for t in self.traces) / len(self.traces),
        }
```

### 4.6 Execution Tracer (Day 9-11)

**File: `src/tcoml/tracing/execution_tracer.py`**

```python
import time
from typing import Any, Dict, List, Optional
from .trace_models import ExecutionTrace, ToolCall
from ..agent.tool_registry import ToolRegistry


class ExecutionTracer:
    """Captures execution traces of agent task attempts"""

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.current_trace: Optional[ExecutionTrace] = None

    def start_trace(self, task_id: str, task_description: str, system_prompt: str) -> None:
        """Start a new execution trace"""
        self.current_trace = ExecutionTrace(
            task_id=task_id,
            task_description=task_description,
            system_prompt=system_prompt
        )
        self.start_time = time.time()

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute and record a tool call.

        Returns the result of the tool execution.
        """
        if not self.current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")

        start = time.time()

        # Execute tool
        tool = self.tool_registry.get(tool_name)
        result = tool.execute(**arguments)

        duration_ms = (time.time() - start) * 1000

        # Record the call
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result.get("result"),
            error=result.get("error"),
            success=result["success"],
            duration_ms=duration_ms
        )

        self.current_trace.add_tool_call(tool_call)

        return result

    def end_trace(self, final_output: Any, task_success: bool, error: Optional[str] = None) -> ExecutionTrace:
        """
        End the current trace and return it.

        Args:
            final_output: The final output from the agent
            task_success: Whether the task was successful
            error: Optional error message if task failed

        Returns:
            The completed ExecutionTrace
        """
        if not self.current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")

        self.current_trace.final_output = final_output
        self.current_trace.task_success = task_success
        self.current_trace.error = error
        self.current_trace.total_time_ms = (time.time() - self.start_time) * 1000

        trace = self.current_trace
        self.current_trace = None

        return trace

    def get_trace_summary(self, trace: ExecutionTrace) -> str:
        """Generate human-readable summary of a trace"""
        lines = [
            f"Task: {trace.task_description}",
            f"Success: {trace.task_success}",
            f"Total Time: {trace.total_time_ms:.2f}ms",
            f"Tool Calls: {len(trace.tool_calls)}",
            "",
            "Execution Steps:"
        ]

        for i, tc in enumerate(trace.tool_calls, 1):
            status = "✓" if tc.success else "✗"
            lines.append(f"  {i}. {status} {tc.tool_name}({tc.arguments})")
            if tc.success:
                lines.append(f"     → {tc.result}")
            else:
                lines.append(f"     → ERROR: {tc.error}")

        lines.append(f"\nFinal Output: {trace.final_output}")

        return "\n".join(lines)
```

### 4.7 LLM Base Client (Day 11-14)

**File: `src/tcoml/llm/base_client.py`**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    """Represents a message in a conversation"""
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Response from LLM"""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = {}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools (function calling)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and optional tool calls
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
```

**File: `src/tcoml/llm/openai_client.py`**

```python
import os
from typing import Any, Dict, List, Optional
import openai
from .base_client import BaseLLMClient, Message, LLMResponse


class OpenAIClient(BaseLLMClient):
    """
    OpenAI LLM client.

    IMPORTANT: Verify current API at implementation time.
    This may be using Response API, Chat Completions API, or newer APIs.
    Check https://platform.openai.com/docs for latest specifications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (check current available models)
            base_url: Optional custom base URL
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

        # Verify current model at implementation time
        # May be gpt-5.x, gpt-6, or newer
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate completion using current OpenAI API"""

        # Convert messages
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add tools if provided (function calling)
        if tools:
            kwargs["tools"] = [{"type": "function", "function": tool} for tool in tools]

        # Make API call
        # NOTE: Verify API method at implementation time
        # May be chat.completions.create() or newer method
        response = self.client.chat.completions.create(**kwargs)

        # Parse response
        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract tool calls if present
        tool_calls = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            import tiktoken
            # Verify encoding at implementation time
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Rough approximation if tiktoken fails
            return len(text) // 4
```

**File: `src/tcoml/llm/anthropic_client.py`**

```python
import os
from typing import Any, Dict, List, Optional
import anthropic
from .base_client import BaseLLMClient, Message, LLMResponse


class AnthropicClient(BaseLLMClient):
    """
    Anthropic LLM client.

    IMPORTANT: Verify current API at implementation time.
    Check https://docs.anthropic.com for latest specifications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (verify current available models)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        # Verify current model at implementation time
        # May be claude-4.x, claude-5.x, or newer
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate completion using Anthropic API"""

        # Extract system message if present
        system_msg = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_msg:
            kwargs["system"] = system_msg

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools

        # Make API call
        response = self.client.messages.create(**kwargs)

        # Parse response
        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        )

    def count_tokens(self, text: str) -> int:
        """Approximate token count for Claude"""
        # Claude uses different tokenization, approximate
        return len(text) // 4
```

**Phase 1 Complete Checkpoint**:
```python
from tcoml.llm.openai_client import OpenAIClient
from tcoml.llm.base_client import Message
from tcoml.tracing.execution_tracer import ExecutionTracer
from tcoml.agent.tool_registry import ToolRegistry

# Initialize components
registry = ToolRegistry()
tracer = ExecutionTracer(registry)
llm = OpenAIClient()

# Test LLM
response = llm.complete([
    Message(role="user", content="What is 2+2?")
])
print(f"LLM response: {response.content}")

# Test tracing
tracer.start_trace("test_001", "Calculate 10+20", "You are a helpful assistant")
result = tracer.record_tool_call("calculator", {"operation": "add", "a": 10, "b": 20})
trace = tracer.end_trace(final_output=result["result"], task_success=True)
print(f"Trace: {tracer.get_trace_summary(trace)}")
```

---

## 5. Phase 2: Feedback Loop (Week 3)

### 5.1 Tool Call Agent (Day 15-17)

**File: `src/tcoml/agent/tool_call_agent.py`**

```python
import json
from typing import Any, Dict, List, Optional
from ..llm.base_client import BaseLLMClient, Message
from ..tracing.execution_tracer import ExecutionTracer
from ..tracing.trace_models import ExecutionTrace
from ..benchmarks.task_models import Task
from .tool_registry import ToolRegistry


class ToolCallAgent:
    """
    Agent that uses LLM to solve tasks via tool calling.

    The agent's behavior is controlled by a system prompt that
    will be optimized via meta-learning.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        """
        Initialize agent.

        Args:
            llm_client: LLM client for generating responses
            tool_registry: Registry of available tools
            system_prompt: System prompt controlling tool usage behavior
            max_iterations: Maximum tool call iterations
            verbose: Whether to print execution details
        """
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose

    def solve_task(self, task: Task) -> ExecutionTrace:
        """
        Solve a task using tools.

        Returns an ExecutionTrace with complete execution history.
        """
        tracer = ExecutionTracer(self.tool_registry)
        tracer.start_trace(task.id, task.description, self.system_prompt)

        # Build tool schemas for LLM
        tool_schemas = [
            self.tool_registry.get(tool_name).to_openai_function()
            for tool_name in task.required_tools
        ]

        # Initialize conversation
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task.description)
        ]

        final_output = None
        error = None

        try:
            for iteration in range(self.max_iterations):
                if self.verbose:
                    print(f"Iteration {iteration + 1}/{self.max_iterations}")

                # Get LLM response
                response = self.llm.complete(
                    messages=messages,
                    tools=tool_schemas,
                    temperature=0.0
                )

                # Check if LLM wants to call tools
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]

                        # Parse arguments (may be JSON string)
                        args = tool_call["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)

                        if self.verbose:
                            print(f"  Calling {tool_name}({args})")

                        # Execute tool via tracer
                        result = tracer.record_tool_call(tool_name, args)

                        # Add tool result to conversation
                        if result["success"]:
                            messages.append(Message(
                                role="assistant",
                                content=f"Called {tool_name}, got result: {result['result']}"
                            ))
                        else:
                            messages.append(Message(
                                role="assistant",
                                content=f"Tool call failed: {result['error']}"
                            ))

                # Check if done (no tool calls or has final answer)
                if not response.tool_calls and response.content:
                    final_output = response.content
                    break

                # Add assistant response to conversation
                if response.content:
                    messages.append(Message(role="assistant", content=response.content))

            # Evaluate success
            task_success = task.evaluate(final_output) if final_output else False

        except Exception as e:
            error = str(e)
            task_success = False
            final_output = None

        return tracer.end_trace(final_output, task_success, error)

    def solve_tasks(self, tasks: List[Task]) -> List[ExecutionTrace]:
        """Solve multiple tasks and return traces"""
        traces = []
        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n=== Task {i+1}/{len(tasks)}: {task.id} ===")
            trace = self.solve_task(task)
            traces.append(trace)
            if self.verbose:
                print(f"Success: {trace.task_success}")
        return traces
```

### 5.2 Feedback Generator (Day 17-19)

**File: `src/tcoml/feedback/feedback_generator.py`**

```python
from typing import List, Dict, Any
from pydantic import BaseModel
from ..llm.base_client import BaseLLMClient, Message
from ..tracing.trace_models import ExecutionTrace
from ..benchmarks.task_models import Task


class Feedback(BaseModel):
    """Feedback on a single trace"""
    trace_id: str
    task_id: str
    success: bool
    critique: str
    suggestions: List[str]
    what_went_wrong: str = ""
    what_went_right: str = ""


class FeedbackGenerator:
    """
    Generates natural language feedback on execution traces.

    This is the "Evaluator" component in the Arize-ai framework.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def generate_feedback(
        self,
        trace: ExecutionTrace,
        task: Task
    ) -> Feedback:
        """
        Generate detailed feedback for a single trace.

        Args:
            trace: Execution trace to analyze
            task: The task that was attempted

        Returns:
            Feedback with critique and suggestions
        """

        # Build analysis prompt
        prompt = self._build_feedback_prompt(trace, task)

        # Get LLM analysis
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse response into structured feedback
        critique_text = response.content

        # Extract suggestions (simple parsing)
        suggestions = self._extract_suggestions(critique_text)

        # Determine what went wrong/right
        what_wrong = self._extract_section(critique_text, "What went wrong")
        what_right = self._extract_section(critique_text, "What went right")

        return Feedback(
            trace_id=f"{trace.task_id}_trace",
            task_id=trace.task_id,
            success=trace.task_success,
            critique=critique_text,
            suggestions=suggestions,
            what_went_wrong=what_wrong,
            what_went_right=what_right
        )

    def _build_feedback_prompt(self, trace: ExecutionTrace, task: Task) -> str:
        """Build prompt for feedback generation"""

        # Format tool calls
        tool_call_desc = []
        for i, tc in enumerate(trace.tool_calls, 1):
            status = "SUCCESS" if tc.success else "FAILED"
            tool_call_desc.append(
                f"{i}. {tc.tool_name}({tc.arguments}) → {status}: {tc.result or tc.error}"
            )
        tool_calls_str = "\n".join(tool_call_desc)

        # Extract available tools from task
        available_tools = ", ".join(task.required_tools)

        prompt = f"""You are an expert at analyzing AI agent tool usage patterns.

Analyze the following task execution and provide detailed feedback.

TASK: {trace.task_description}
SUCCESS: {trace.task_success}
AVAILABLE TOOLS: {available_tools}

EXECUTION TRACE:
{tool_calls_str}

FINAL OUTPUT: {trace.final_output}
EXPECTED: {task.ground_truth if task.ground_truth else "See success criteria"}

Please analyze this execution and provide:

1. **What went wrong** (if task failed):
   - Which tool calls were incorrect?
   - What was the reasoning error?
   - Were there missing steps?

2. **What went right** (if task succeeded):
   - Which strategies worked well?
   - Was the approach efficient?

3. **Specific suggestions** for improving the system prompt:
   - How should tool usage instructions be modified?
   - What patterns should be encouraged/discouraged?
   - Are there better strategies?

Format your response with clear sections:
- What went wrong: ...
- What went right: ...
- Suggestions:
  - Suggestion 1
  - Suggestion 2
  - ...
"""
        return prompt

    def _extract_suggestions(self, text: str) -> List[str]:
        """Extract bullet-pointed suggestions from feedback"""
        suggestions = []
        in_suggestions = False

        for line in text.split("\n"):
            line = line.strip()
            if "suggestions:" in line.lower():
                in_suggestions = True
                continue

            if in_suggestions:
                # Check for bullet points
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    suggestion = line.lstrip("-•* ").strip()
                    if suggestion:
                        suggestions.append(suggestion)
                elif line and not line.endswith(":"):
                    # Stop at next section
                    break

        return suggestions

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from feedback text"""
        lines = text.split("\n")
        content = []
        in_section = False

        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                # Check if content is on same line after ":"
                if ":" in line:
                    content_part = line.split(":", 1)[1].strip()
                    if content_part:
                        content.append(content_part)
                continue

            if in_section:
                # Stop at next section header (ends with :)
                if line.strip().endswith(":") and len(line.strip()) < 50:
                    break
                if line.strip():
                    content.append(line.strip())

        return " ".join(content)

    def generate_batch_feedback(
        self,
        traces: List[ExecutionTrace],
        tasks: Dict[str, Task]
    ) -> List[Feedback]:
        """Generate feedback for multiple traces"""
        feedbacks = []
        for trace in traces:
            task = tasks.get(trace.task_id)
            if task:
                feedback = self.generate_feedback(trace, task)
                feedbacks.append(feedback)
        return feedbacks
```

### 5.3 Feedback Aggregator (Day 19-21)

**File: `src/tcoml/feedback/feedback_aggregator.py`**

```python
from typing import List, Dict, Any
from collections import Counter
from pydantic import BaseModel
from ..llm.base_client import BaseLLMClient, Message
from .feedback_generator import Feedback


class AggregatedFeedback(BaseModel):
    """Aggregated feedback across multiple traces"""
    total_traces: int
    successful_traces: int
    failed_traces: int
    common_failure_patterns: List[str]
    common_success_patterns: List[str]
    top_suggestions: List[str]
    summary: str


class FeedbackAggregator:
    """
    Aggregates feedback from multiple traces to identify patterns.

    This helps the meta-prompter focus on systemic issues rather
    than task-specific problems.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def aggregate(self, feedbacks: List[Feedback]) -> AggregatedFeedback:
        """
        Aggregate multiple feedback items into patterns.

        Args:
            feedbacks: List of individual feedback items

        Returns:
            Aggregated feedback with common patterns
        """
        if not feedbacks:
            return AggregatedFeedback(
                total_traces=0,
                successful_traces=0,
                failed_traces=0,
                common_failure_patterns=[],
                common_success_patterns=[],
                top_suggestions=[],
                summary="No feedback to aggregate"
            )

        # Basic statistics
        total = len(feedbacks)
        successful = sum(1 for f in feedbacks if f.success)
        failed = total - successful

        # Collect all suggestions
        all_suggestions = []
        for f in feedbacks:
            all_suggestions.extend(f.suggestions)

        # Find most common suggestions (simple frequency)
        suggestion_counts = Counter(all_suggestions)
        top_suggestions = [s for s, count in suggestion_counts.most_common(10)]

        # Analyze failure patterns
        failure_feedbacks = [f for f in feedbacks if not f.success]
        failure_patterns = self._identify_patterns(
            [f.what_went_wrong for f in failure_feedbacks],
            "failure"
        )

        # Analyze success patterns
        success_feedbacks = [f for f in feedbacks if f.success]
        success_patterns = self._identify_patterns(
            [f.what_went_right for f in success_feedbacks],
            "success"
        )

        # Generate summary
        summary = self._generate_summary(
            total, successful, failed,
            failure_patterns, success_patterns, top_suggestions
        )

        return AggregatedFeedback(
            total_traces=total,
            successful_traces=successful,
            failed_traces=failed,
            common_failure_patterns=failure_patterns,
            common_success_patterns=success_patterns,
            top_suggestions=top_suggestions,
            summary=summary
        )

    def _identify_patterns(self, texts: List[str], pattern_type: str) -> List[str]:
        """Use LLM to identify common patterns in feedback texts"""
        if not texts:
            return []

        # Combine all texts
        combined = "\n\n".join([f"- {t}" for t in texts if t])

        if not combined:
            return []

        prompt = f"""Analyze the following {pattern_type} descriptions and identify 3-5 common patterns.

{combined}

Extract the most common {pattern_type} patterns. Be specific and concise.
Return as a simple list:
- Pattern 1
- Pattern 2
- ...
"""

        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=500
        )

        # Parse patterns from response
        patterns = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                pattern = line.lstrip("-•* ").strip()
                if pattern:
                    patterns.append(pattern)

        return patterns[:5]  # Top 5

    def _generate_summary(
        self,
        total: int,
        successful: int,
        failed: int,
        failure_patterns: List[str],
        success_patterns: List[str],
        top_suggestions: List[str]
    ) -> str:
        """Generate natural language summary"""

        success_rate = successful / total if total > 0 else 0

        summary_parts = [
            f"Analyzed {total} task executions ({successful} successful, {failed} failed).",
            f"Success rate: {success_rate:.1%}",
        ]

        if failure_patterns:
            summary_parts.append("\nCommon failure patterns:")
            summary_parts.extend([f"- {p}" for p in failure_patterns[:3]])

        if success_patterns:
            summary_parts.append("\nCommon success patterns:")
            summary_parts.extend([f"- {p}" for p in success_patterns[:3]])

        if top_suggestions:
            summary_parts.append("\nTop suggestions for improvement:")
            summary_parts.extend([f"- {s}" for s in top_suggestions[:3]])

        return "\n".join(summary_parts)
```

**Phase 2 Complete Checkpoint**:
```python
from tcoml.agent.tool_call_agent import ToolCallAgent
from tcoml.feedback.feedback_generator import FeedbackGenerator
from tcoml.feedback.feedback_aggregator import FeedbackAggregator
from tcoml.llm.openai_client import OpenAIClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

# Setup
llm = OpenAIClient()
registry = ToolRegistry()
tasks = SyntheticTaskGenerator().generate_calculator_tasks(5)

# Create agent with baseline prompt
baseline_prompt = """You are a helpful assistant that solves tasks using available tools.
Use tools to complete the task accurately."""

agent = ToolCallAgent(llm, registry, baseline_prompt, verbose=True)

# Solve tasks and collect traces
traces = agent.solve_tasks(tasks)

# Generate feedback
feedback_gen = FeedbackGenerator(llm)
task_dict = {t.id: t for t in tasks}
feedbacks = feedback_gen.generate_batch_feedback(traces, task_dict)

# Aggregate feedback
aggregator = FeedbackAggregator(llm)
aggregated = aggregator.aggregate(feedbacks)
print(aggregated.summary)
```

---

## 6. Phase 3: Meta-Prompting (Week 4)

### 6.1 Meta-Prompter (Day 22-25)

**File: `src/tcoml/optimization/meta_prompter.py`**

```python
from typing import List, Optional
from pydantic import BaseModel
from ..llm.base_client import BaseLLMClient, Message
from ..feedback.feedback_aggregator import AggregatedFeedback


class PromptUpdate(BaseModel):
    """Result of meta-prompting"""
    new_prompt: str
    changes_made: List[str]
    rationale: str
    iteration: int


class MetaPrompter:
    """
    Uses LLM to improve system prompts based on aggregated feedback.

    This is the "Optimizer" component in the Arize-ai framework.
    """

    def __init__(self, llm_client: BaseLLMClient, verbose: bool = False):
        self.llm = llm_client
        self.verbose = verbose

    def improve_prompt(
        self,
        current_prompt: str,
        feedback: AggregatedFeedback,
        iteration: int,
        previous_prompts: Optional[List[str]] = None
    ) -> PromptUpdate:
        """
        Generate improved system prompt based on feedback.

        Args:
            current_prompt: Current system prompt
            feedback: Aggregated feedback from task executions
            iteration: Current iteration number
            previous_prompts: History of previous prompts (to avoid cycles)

        Returns:
            PromptUpdate with new prompt and explanation
        """

        # Build meta-prompting request
        meta_prompt = self._build_meta_prompt(
            current_prompt,
            feedback,
            iteration,
            previous_prompts
        )

        if self.verbose:
            print(f"\n=== Meta-Prompting (Iteration {iteration}) ===")
            print(f"Current success rate: {feedback.successful_traces}/{feedback.total_traces}")

        # Get improved prompt from LLM
        response = self.llm.complete(
            messages=[Message(role="user", content=meta_prompt)],
            temperature=0.7,  # Higher temperature for creativity
            max_tokens=2000
        )

        # Parse response
        new_prompt, changes, rationale = self._parse_meta_response(response.content)

        if self.verbose:
            print(f"\nChanges made:")
            for change in changes:
                print(f"  - {change}")

        return PromptUpdate(
            new_prompt=new_prompt,
            changes_made=changes,
            rationale=rationale,
            iteration=iteration
        )

    def _build_meta_prompt(
        self,
        current_prompt: str,
        feedback: AggregatedFeedback,
        iteration: int,
        previous_prompts: Optional[List[str]] = None
    ) -> str:
        """Build the meta-prompting request"""

        prompt_parts = [
            "You are an expert at optimizing system prompts for AI agents that use tools.",
            "",
            "**Current Task**: Improve a system prompt that controls how an agent uses tools to solve tasks.",
            "",
            f"**Current System Prompt** (Iteration {iteration}):",
            "```",
            current_prompt,
            "```",
            "",
            "**Performance Feedback**:",
            feedback.summary,
            "",
        ]

        if feedback.common_failure_patterns:
            prompt_parts.append("**Common Failure Patterns**:")
            for pattern in feedback.common_failure_patterns:
                prompt_parts.append(f"- {pattern}")
            prompt_parts.append("")

        if feedback.common_success_patterns:
            prompt_parts.append("**Common Success Patterns**:")
            for pattern in feedback.common_success_patterns:
                prompt_parts.append(f"- {pattern}")
            prompt_parts.append("")

        if feedback.top_suggestions:
            prompt_parts.append("**Top Suggestions**:")
            for suggestion in feedback.top_suggestions[:5]:
                prompt_parts.append(f"- {suggestion}")
            prompt_parts.append("")

        prompt_parts.extend([
            "**Your Task**:",
            "1. Analyze the current prompt and feedback",
            "2. Identify specific weaknesses in the current prompt",
            "3. Generate an IMPROVED system prompt that addresses the issues",
            "4. The new prompt should:",
            "   - Provide clearer instructions for tool usage",
            "   - Address common failure patterns",
            "   - Reinforce successful strategies",
            "   - Be specific and actionable",
            "   - Avoid being overly verbose",
            "",
            "**Output Format**:",
            "NEW_PROMPT:",
            "```",
            "[Your improved system prompt here]",
            "```",
            "",
            "CHANGES_MADE:",
            "- Change 1",
            "- Change 2",
            "- ...",
            "",
            "RATIONALE:",
            "[Explanation of why these changes will improve performance]",
        ])

        # Add history warning to avoid cycles
        if previous_prompts and len(previous_prompts) > 2:
            prompt_parts.extend([
                "",
                "**Important**: Avoid simply reverting to previous versions. Make genuine improvements.",
            ])

        return "\n".join(prompt_parts)

    def _parse_meta_response(self, response_text: str) -> tuple[str, List[str], str]:
        """
        Parse the meta-prompter's response.

        Returns:
            (new_prompt, changes_made, rationale)
        """
        lines = response_text.split("\n")

        new_prompt_lines = []
        changes = []
        rationale_lines = []

        section = None
        in_code_block = False

        for line in lines:
            # Detect sections
            if "NEW_PROMPT:" in line.upper():
                section = "prompt"
                continue
            elif "CHANGES_MADE:" in line.upper() or "CHANGES:" in line.upper():
                section = "changes"
                continue
            elif "RATIONALE:" in line.upper():
                section = "rationale"
                continue

            # Handle code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            # Collect content based on section
            if section == "prompt" and (in_code_block or line.strip()):
                new_prompt_lines.append(line)
            elif section == "changes":
                line = line.strip()
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    change = line.lstrip("-•* ").strip()
                    if change:
                        changes.append(change)
            elif section == "rationale" and line.strip():
                rationale_lines.append(line.strip())

        new_prompt = "\n".join(new_prompt_lines).strip()
        rationale = " ".join(rationale_lines)

        # Fallback if parsing failed
        if not new_prompt:
            new_prompt = response_text.strip()
        if not changes:
            changes = ["Prompt updated based on feedback"]
        if not rationale:
            rationale = "Improvements based on execution feedback"

        return new_prompt, changes, rationale
```

### 6.2 Optimization Loop (Day 25-28)

**File: `src/tcoml/optimization/optimization_loop.py`**

```python
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from ..agent.tool_call_agent import ToolCallAgent
from ..agent.tool_registry import ToolRegistry
from ..benchmarks.task_models import Task
from ..feedback.feedback_generator import FeedbackGenerator
from ..feedback.feedback_aggregator import FeedbackAggregator
from ..llm.base_client import BaseLLMClient
from ..tracing.trace_models import ExecutionTrace, TraceDataset
from .meta_prompter import MetaPrompter, PromptUpdate


class OptimizationResult(BaseModel):
    """Results from optimization loop"""
    iterations: int
    initial_success_rate: float
    final_success_rate: float
    best_success_rate: float
    best_iteration: int
    best_prompt: str
    prompt_history: List[str]
    performance_history: List[float]
    converged: bool
    convergence_iteration: Optional[int] = None


class OptimizationLoop:
    """
    Main optimization loop orchestrating the meta-learning process.

    Implements the full Arize-ai inspired workflow:
    1. Train/Test Split
    2. Execute tasks with current prompt
    3. Generate feedback
    4. Aggregate feedback
    5. Meta-prompt to improve
    6. Re-evaluate
    7. Repeat
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        output_dir: Path,
        max_iterations: int = 10,
        convergence_threshold: float = 0.02,
        train_test_split: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize optimization loop.

        Args:
            llm_client: LLM client for agent and meta-prompting
            tool_registry: Available tools
            output_dir: Directory to save results
            max_iterations: Maximum optimization iterations
            convergence_threshold: Success rate change threshold for convergence
            train_test_split: Fraction of tasks for training (rest for testing)
            verbose: Print progress
        """
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.train_test_split = train_test_split
        self.verbose = verbose

        # Initialize components
        self.feedback_generator = FeedbackGenerator(llm_client)
        self.feedback_aggregator = FeedbackAggregator(llm_client)
        self.meta_prompter = MetaPrompter(llm_client, verbose=verbose)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir(exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "feedback").mkdir(exist_ok=True)

    def run(
        self,
        tasks: List[Task],
        initial_prompt: str
    ) -> OptimizationResult:
        """
        Run the full optimization loop.

        Args:
            tasks: List of tasks for training and testing
            initial_prompt: Starting system prompt

        Returns:
            OptimizationResult with performance and prompts
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Optimization Loop")
            print(f"{'='*70}")
            print(f"Total tasks: {len(tasks)}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Convergence threshold: {self.convergence_threshold}")

        # Step 1: Train/Test Split
        train_tasks, test_tasks = self._split_tasks(tasks)

        if self.verbose:
            print(f"\nTrain tasks: {len(train_tasks)}")
            print(f"Test tasks: {len(test_tasks)}")

        # Initialize tracking
        current_prompt = initial_prompt
        prompt_history = [initial_prompt]
        performance_history = []
        best_success_rate = 0.0
        best_iteration = 0
        best_prompt = initial_prompt
        converged = False
        convergence_iteration = None

        # Save initial prompt
        self._save_prompt(initial_prompt, 0)

        # Main optimization loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*70}")

            # Step 2: Execute tasks with current prompt
            agent = ToolCallAgent(
                llm_client=self.llm,
                tool_registry=self.tool_registry,
                system_prompt=current_prompt,
                verbose=False
            )

            traces = agent.solve_tasks(train_tasks)

            # Calculate success rate
            success_count = sum(1 for t in traces if t.task_success)
            success_rate = success_count / len(train_tasks)
            performance_history.append(success_rate)

            if self.verbose:
                print(f"\nTrain Success Rate: {success_rate:.1%} ({success_count}/{len(train_tasks)})")

            # Track best
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_iteration = iteration
                best_prompt = current_prompt

            # Save traces
            self._save_traces(traces, iteration)

            # Step 3 & 4: Generate and aggregate feedback
            task_dict = {t.id: t for t in train_tasks}
            feedbacks = self.feedback_generator.generate_batch_feedback(traces, task_dict)
            aggregated_feedback = self.feedback_aggregator.aggregate(feedbacks)

            # Save feedback
            self._save_feedback(aggregated_feedback, iteration)

            # Check convergence (if not first iteration)
            if iteration > 0:
                improvement = success_rate - performance_history[-2]
                if abs(improvement) < self.convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    if self.verbose:
                        print(f"\n✓ Converged! Improvement {improvement:.1%} < threshold {self.convergence_threshold:.1%}")
                    break

            # Step 5: Meta-prompt to generate improved prompt
            if iteration < self.max_iterations - 1:  # Don't update on last iteration
                prompt_update = self.meta_prompter.improve_prompt(
                    current_prompt=current_prompt,
                    feedback=aggregated_feedback,
                    iteration=iteration + 1,
                    previous_prompts=prompt_history
                )

                current_prompt = prompt_update.new_prompt
                prompt_history.append(current_prompt)

                # Save new prompt
                self._save_prompt(current_prompt, iteration + 1)

                if self.verbose:
                    print(f"\nPrompt updated for next iteration")

        # Step 6: Final evaluation on test set
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Final Evaluation on Test Set")
            print(f"{'='*70}")

        test_agent = ToolCallAgent(
            llm_client=self.llm,
            tool_registry=self.tool_registry,
            system_prompt=best_prompt,
            verbose=False
        )

        test_traces = test_agent.solve_tasks(test_tasks)
        test_success_count = sum(1 for t in test_traces if t.task_success)
        final_success_rate = test_success_count / len(test_tasks)

        if self.verbose:
            print(f"\nTest Success Rate: {final_success_rate:.1%} ({test_success_count}/{len(test_tasks)})")
            print(f"Best Train Success Rate: {best_success_rate:.1%} (Iteration {best_iteration})")

        # Save test traces
        self._save_traces(test_traces, iteration=-1, prefix="test")

        # Create result
        result = OptimizationResult(
            iterations=iteration + 1,
            initial_success_rate=performance_history[0],
            final_success_rate=final_success_rate,
            best_success_rate=best_success_rate,
            best_iteration=best_iteration,
            best_prompt=best_prompt,
            prompt_history=prompt_history,
            performance_history=performance_history,
            converged=converged,
            convergence_iteration=convergence_iteration
        )

        # Save final result
        self._save_result(result)

        return result

    def _split_tasks(self, tasks: List[Task]) -> tuple[List[Task], List[Task]]:
        """Split tasks into train and test sets"""
        import random

        # Shuffle with fixed seed for reproducibility
        shuffled = tasks.copy()
        random.seed(42)
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * self.train_test_split)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        return train, test

    def _save_prompt(self, prompt: str, iteration: int) -> None:
        """Save prompt to file"""
        filename = self.output_dir / "prompts" / f"prompt_iter_{iteration:03d}.txt"
        filename.write_text(prompt)

    def _save_traces(self, traces: List[ExecutionTrace], iteration: int, prefix: str = "train") -> None:
        """Save traces to JSON"""
        dataset = TraceDataset(traces=traces)
        filename = self.output_dir / "traces" / f"{prefix}_traces_iter_{iteration:03d}.json"
        filename.write_text(dataset.model_dump_json(indent=2))

    def _save_feedback(self, feedback: AggregatedFeedback, iteration: int) -> None:
        """Save aggregated feedback"""
        filename = self.output_dir / "feedback" / f"feedback_iter_{iteration:03d}.json"
        filename.write_text(feedback.model_dump_json(indent=2))

    def _save_result(self, result: OptimizationResult) -> None:
        """Save final optimization result"""
        filename = self.output_dir / "optimization_result.json"
        filename.write_text(result.model_dump_json(indent=2))

        if self.verbose:
            print(f"\n✓ Results saved to {self.output_dir}")
```

**Phase 3 Complete Checkpoint**:
```python
from pathlib import Path
from tcoml.optimization.optimization_loop import OptimizationLoop
from tcoml.llm.openai_client import OpenAIClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

# Setup
llm = OpenAIClient()
registry = ToolRegistry()
tasks = SyntheticTaskGenerator().generate_all(n_per_type=10)  # 30 tasks total

# Initial baseline prompt
initial_prompt = """You are a helpful assistant that can use tools to solve tasks.
When given a task, use the available tools to complete it."""

# Run optimization
loop = OptimizationLoop(
    llm_client=llm,
    tool_registry=registry,
    output_dir=Path("./data/results/experiment_001"),
    max_iterations=5,
    verbose=True
)

result = loop.run(tasks, initial_prompt)

print(f"\nOptimization Complete!")
print(f"Initial: {result.initial_success_rate:.1%}")
print(f"Final: {result.final_success_rate:.1%}")
print(f"Improvement: {result.final_success_rate - result.initial_success_rate:+.1%}")
```

---

## 7. Phase 4: Evaluation (Week 5)

### 7.1 Evaluation Metrics (Day 29-31)

**File: `src/tcoml/evaluation/metrics.py`**

```python
import numpy as np
from typing import List, Dict, Any
from ..tracing.trace_models import ExecutionTrace, TraceDataset


class EvaluationMetrics:
    """Comprehensive evaluation metrics for agent performance"""

    @staticmethod
    def success_rate(traces: List[ExecutionTrace]) -> float:
        """Calculate task success rate"""
        if not traces:
            return 0.0
        return sum(1 for t in traces if t.task_success) / len(traces)

    @staticmethod
    def avg_tool_calls(traces: List[ExecutionTrace]) -> float:
        """Average number of tool calls per task"""
        if not traces:
            return 0.0
        return sum(len(t.tool_calls) for t in traces) / len(traces)

    @staticmethod
    def avg_successful_tool_calls(traces: List[ExecutionTrace]) -> float:
        """Average number of successful tool calls"""
        if not traces:
            return 0.0
        successful = sum(
            sum(1 for tc in t.tool_calls if tc.success)
            for t in traces
        )
        return successful / len(traces)

    @staticmethod
    def tool_call_efficiency(traces: List[ExecutionTrace]) -> float:
        """
        Ratio of successful tool calls to total tool calls.
        Higher is better (less wasted calls).
        """
        total_calls = sum(len(t.tool_calls) for t in traces)
        if total_calls == 0:
            return 0.0

        successful_calls = sum(
            sum(1 for tc in t.tool_calls if tc.success)
            for t in traces
        )
        return successful_calls / total_calls

    @staticmethod
    def first_attempt_success_rate(traces: List[ExecutionTrace]) -> float:
        """
        Percentage of tasks solved without any failed tool calls.
        Measures how often the agent gets it right immediately.
        """
        if not traces:
            return 0.0

        first_attempt = sum(
            1 for t in traces
            if t.task_success and all(tc.success for tc in t.tool_calls)
        )
        return first_attempt / len(traces)

    @staticmethod
    def error_recovery_rate(traces: List[ExecutionTrace]) -> float:
        """
        Percentage of tasks that succeeded despite having failed tool calls.
        Measures agent's ability to recover from errors.
        """
        if not traces:
            return 0.0

        recovered = sum(
            1 for t in traces
            if t.task_success and any(not tc.success for tc in t.tool_calls)
        )

        # Denominator: tasks that had at least one error
        tasks_with_errors = sum(
            1 for t in traces
            if any(not tc.success for tc in t.tool_calls)
        )

        if tasks_with_errors == 0:
            return 0.0

        return recovered / tasks_with_errors

    @staticmethod
    def avg_execution_time(traces: List[ExecutionTrace]) -> float:
        """Average execution time in milliseconds"""
        if not traces:
            return 0.0
        return sum(t.total_time_ms for t in traces) / len(traces)

    @staticmethod
    def bootstrap_confidence_interval(
        traces: List[ExecutionTrace],
        metric_fn: callable,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Returns:
            (metric_value, lower_bound, upper_bound)
        """
        metric_value = metric_fn(traces)

        # Bootstrap resampling
        bootstrap_values = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(traces, size=len(traces), replace=True)
            bootstrap_values.append(metric_fn(list(sample)))

        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

        return metric_value, lower, upper

    @staticmethod
    def compare_performance(
        baseline_traces: List[ExecutionTrace],
        optimized_traces: List[ExecutionTrace],
        metric_fn: callable = None
    ) -> Dict[str, Any]:
        """
        Compare performance between baseline and optimized.

        Returns statistical significance test results.
        """
        if metric_fn is None:
            metric_fn = EvaluationMetrics.success_rate

        baseline_value = metric_fn(baseline_traces)
        optimized_value = metric_fn(optimized_traces)
        improvement = optimized_value - baseline_value

        # Permutation test for statistical significance
        combined = baseline_traces + optimized_traces
        n_baseline = len(baseline_traces)

        observed_diff = optimized_value - baseline_value

        # Permutation test
        n_permutations = 1000
        permuted_diffs = []

        for _ in range(n_permutations):
            # Shuffle and split
            shuffled = np.random.permutation(combined)
            perm_baseline = shuffled[:n_baseline]
            perm_optimized = shuffled[n_baseline:]

            perm_diff = metric_fn(list(perm_optimized)) - metric_fn(list(perm_baseline))
            permuted_diffs.append(perm_diff)

        # P-value: proportion of permutations with diff >= observed
        p_value = np.mean([abs(d) >= abs(observed_diff) for d in permuted_diffs])

        return {
            "baseline_value": baseline_value,
            "optimized_value": optimized_value,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_value * 100) if baseline_value > 0 else 0,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    @staticmethod
    def comprehensive_report(traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        return {
            "success_rate": EvaluationMetrics.success_rate(traces),
            "avg_tool_calls": EvaluationMetrics.avg_tool_calls(traces),
            "tool_call_efficiency": EvaluationMetrics.tool_call_efficiency(traces),
            "first_attempt_success": EvaluationMetrics.first_attempt_success_rate(traces),
            "error_recovery_rate": EvaluationMetrics.error_recovery_rate(traces),
            "avg_execution_time_ms": EvaluationMetrics.avg_execution_time(traces),
            "total_traces": len(traces),
        }
```

### 7.2 Experiment Runner (Day 31-35)

**File: `src/scripts/run_experiment.py`**

```python
#!/usr/bin/env python3
"""
Run a complete optimization experiment with evaluation.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from tcoml.llm.openai_client import OpenAIClient
from tcoml.llm.anthropic_client import AnthropicClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator
from tcoml.optimization.optimization_loop import OptimizationLoop
from tcoml.agent.tool_call_agent import ToolCallAgent
from tcoml.evaluation.metrics import EvaluationMetrics


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiment")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", type=str, help="Model name (defaults to env)")
    parser.add_argument("--n-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--max-iter", type=int, default=10, help="Max optimization iterations")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./data/results/experiment_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Initialize LLM client
    if args.provider == "openai":
        llm = OpenAIClient(model=args.model)
    elif args.provider == "anthropic":
        llm = AnthropicClient(model=args.model)

    print(f"Using {args.provider} with model: {llm.model}")

    # Initialize tool registry
    registry = ToolRegistry()

    # Generate tasks
    print(f"\nGenerating {args.n_tasks} synthetic tasks...")
    generator = SyntheticTaskGenerator()
    tasks = generator.generate_all(n_per_type=args.n_tasks // 3)
    print(f"Generated {len(tasks)} tasks")

    # Save tasks
    tasks_file = output_dir / "tasks.json"
    with open(tasks_file, "w") as f:
        task_data = [
            {
                "id": t.id,
                "description": t.description,
                "required_tools": t.required_tools,
                "difficulty": t.difficulty,
                "category": t.category,
                "ground_truth": t.ground_truth,
            }
            for t in tasks
        ]
        json.dump(task_data, f, indent=2)

    # Baseline evaluation
    print("\n" + "="*70)
    print("Baseline Evaluation")
    print("="*70)

    baseline_prompt = """You are a helpful assistant that can use tools to solve tasks.
When given a task, use the available tools to complete it."""

    baseline_agent = ToolCallAgent(llm, registry, baseline_prompt, verbose=False)
    baseline_traces = baseline_agent.solve_tasks(tasks)

    baseline_metrics = EvaluationMetrics.comprehensive_report(baseline_traces)
    print(f"\nBaseline Success Rate: {baseline_metrics['success_rate']:.1%}")
    print(f"Baseline Avg Tool Calls: {baseline_metrics['avg_tool_calls']:.2f}")

    # Save baseline
    baseline_file = output_dir / "baseline_results.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # Run optimization
    print("\n" + "="*70)
    print("Running Optimization")
    print("="*70)

    loop = OptimizationLoop(
        llm_client=llm,
        tool_registry=registry,
        output_dir=output_dir,
        max_iterations=args.max_iter,
        verbose=args.verbose
    )

    result = loop.run(tasks, baseline_prompt)

    # Final comparison
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)

    print(f"\nInitial Success Rate: {result.initial_success_rate:.1%}")
    print(f"Final Success Rate: {result.final_success_rate:.1%}")
    print(f"Improvement: {result.final_success_rate - result.initial_success_rate:+.1%}")
    print(f"Best Iteration: {result.best_iteration}")
    print(f"Converged: {result.converged}")

    # Save summary
    summary = {
        "experiment": {
            "provider": args.provider,
            "model": llm.model,
            "n_tasks": len(tasks),
            "max_iterations": args.max_iter,
            "timestamp": datetime.now().isoformat(),
        },
        "baseline": baseline_metrics,
        "optimization": {
            "iterations": result.iterations,
            "initial_success_rate": result.initial_success_rate,
            "final_success_rate": result.final_success_rate,
            "improvement": result.final_success_rate - result.initial_success_rate,
            "converged": result.converged,
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Experiment complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
```

Make the script executable:
```bash
chmod +x src/scripts/run_experiment.py
```

**Phase 4 Complete - Run Experiment**:
```bash
# Full experiment with 100 tasks, 10 iterations
python src/scripts/run_experiment.py \
    --provider openai \
    --n-tasks 100 \
    --max-iter 10 \
    --verbose

# Results will be in ./data/results/experiment_<timestamp>/
```

---

## 8. Phase 5: Real Benchmarks (Week 6)

**Note**: This phase involves integrating external benchmarks. Implementation details depend on current benchmark versions at implementation time.

### 8.1 Research Current Benchmarks

Before implementing, research:
- **ToolBench**: Check current version and API
- **SWE-Bench Lite**: Verify current format
- **API-Bank**: Check availability
- Alternative benchmarks that may have emerged

### 8.2 Benchmark Adapter Interface

**File: `src/tcoml/benchmarks/benchmark_adapter.py`**

```python
from abc import ABC, abstractmethod
from typing import List
from .task_models import Task


class BenchmarkAdapter(ABC):
    """Abstract interface for external benchmarks"""

    @abstractmethod
    def load_tasks(self, n: int = None) -> List[Task]:
        """Load tasks from benchmark"""
        pass

    @abstractmethod
    def evaluate_result(self, task_id: str, output: Any) -> bool:
        """Evaluate if output is correct for task"""
        pass
```

**Placeholder Implementation** (update when integrating real benchmark):
```python
class ToolBenchAdapter(BenchmarkAdapter):
    """
    Adapter for ToolBench dataset.

    IMPORTANT: Verify current ToolBench version and API before implementing.
    Check: https://github.com/OpenBMB/ToolBench
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        # TODO: Load ToolBench data

    def load_tasks(self, n: int = None) -> List[Task]:
        # TODO: Convert ToolBench format to Task objects
        raise NotImplementedError("Update with current ToolBench format")

    def evaluate_result(self, task_id: str, output: Any) -> bool:
        # TODO: Use ToolBench evaluation logic
        raise NotImplementedError("Update with current evaluation method")
```

---

## 9. Phase 6: Analysis & Documentation (Week 7-8)

### 9.1 Results Analysis Notebook

**File: `notebooks/03_analyze_results.ipynb`**

Create a Jupyter notebook with:

```python
# Cell 1: Load results
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

experiment_dir = Path("../data/results/experiment_20250122_140523")

# Load optimization result
with open(experiment_dir / "optimization_result.json") as f:
    result = json.load(f)

# Cell 2: Performance over iterations
plt.figure(figsize=(10, 6))
plt.plot(result["performance_history"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Success Rate")
plt.title("Optimization Progress")
plt.grid(True)
plt.show()

# Cell 3: Analyze prompt changes
prompt_files = sorted((experiment_dir / "prompts").glob("*.txt"))
prompts = [f.read_text() for f in prompt_files]

print(f"Prompt evolution ({len(prompts)} versions):\n")
for i, prompt in enumerate(prompts[:3]):  # Show first 3
    print(f"=== Iteration {i} ===")
    print(prompt[:200] + "...")
    print()

# Cell 4: Detailed metrics comparison
# Load baseline and final traces, compare comprehensive metrics

# Cell 5: Qualitative analysis
# Extract common patterns from feedback files
```

### 9.2 Create EXPERIMENTS.md

Document all findings in `EXPERIMENTS.md`:

```markdown
# Experiment Results

## Experiment 1: Baseline Optimization

**Date**: 2025-01-22
**Tasks**: 100 synthetic tasks (calculator, string, multi-step)
**Model**: [Current model name]

### Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Success Rate | 45.0% | 68.0% | +23.0% |
| Avg Tool Calls | 3.2 | 2.8 | -12.5% |
| Tool Efficiency | 75% | 89% | +14% |
| First Attempt | 32% | 51% | +19% |

### Learned Patterns

1. **Tool Selection**: Optimizer learned to...
2. **Error Recovery**: Improved by...
3. **Efficiency**: Reduced unnecessary calls by...

### Prompt Evolution

[Analysis of how prompts changed over iterations]

### Insights

[What we learned about tool usage patterns]
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**File: `tests/unit/test_tracer.py`**

```python
import pytest
from tcoml.tracing.execution_tracer import ExecutionTracer
from tcoml.agent.tool_registry import ToolRegistry


def test_tracer_records_successful_call():
    registry = ToolRegistry()
    tracer = ExecutionTracer(registry)

    tracer.start_trace("test_001", "Test task", "Test prompt")
    result = tracer.record_tool_call("calculator", {"operation": "add", "a": 5, "b": 3})

    assert result["success"]
    assert result["result"] == 8

    trace = tracer.end_trace(final_output=8, task_success=True)

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].success
    assert trace.task_success


def test_tracer_records_failed_call():
    registry = ToolRegistry()
    tracer = ExecutionTracer(registry)

    tracer.start_trace("test_002", "Test task", "Test prompt")
    result = tracer.record_tool_call("calculator", {"operation": "divide", "a": 10, "b": 0})

    assert not result["success"]
    assert "error" in result

    trace = tracer.end_trace(final_output=None, task_success=False)

    assert len(trace.tool_calls) == 1
    assert not trace.tool_calls[0].success
```

**File: `tests/unit/test_metrics.py`**

```python
import pytest
from tcoml.evaluation.metrics import EvaluationMetrics
from tcoml.tracing.trace_models import ExecutionTrace, ToolCall


def test_success_rate():
    traces = [
        ExecutionTrace(
            task_id="1",
            task_description="Task 1",
            system_prompt="Prompt",
            task_success=True
        ),
        ExecutionTrace(
            task_id="2",
            task_description="Task 2",
            system_prompt="Prompt",
            task_success=False
        ),
    ]

    rate = EvaluationMetrics.success_rate(traces)
    assert rate == 0.5


def test_tool_call_efficiency():
    trace1 = ExecutionTrace(
        task_id="1",
        task_description="Task 1",
        system_prompt="Prompt",
        task_success=True
    )
    trace1.tool_calls = [
        ToolCall(tool_name="calc", arguments={}, success=True),
        ToolCall(tool_name="calc", arguments={}, success=True),
    ]

    trace2 = ExecutionTrace(
        task_id="2",
        task_description="Task 2",
        system_prompt="Prompt",
        task_success=False
    )
    trace2.tool_calls = [
        ToolCall(tool_name="calc", arguments={}, success=True),
        ToolCall(tool_name="calc", arguments={}, success=False),
    ]

    efficiency = EvaluationMetrics.tool_call_efficiency([trace1, trace2])
    assert efficiency == 0.75  # 3 success out of 4 total
```

Run tests:
```bash
pytest tests/unit -v
```

---

## 11. Configuration Management

### 11.1 Configuration Files

**File: `configs/default.yaml`**

```yaml
# Default configuration for optimization experiments

llm:
  provider: "openai"  # openai, anthropic, local
  model: null  # null = use env default
  temperature: 0.0
  max_tokens: 1000

optimization:
  max_iterations: 10
  convergence_threshold: 0.02
  train_test_split: 0.7

tasks:
  total_tasks: 100
  difficulty_distribution:
    level_1: 0.3
    level_2: 0.3
    level_3: 0.2
    level_4: 0.1
    level_5: 0.1

agent:
  max_tool_iterations: 10
  verbose: false

evaluation:
  bootstrap_samples: 1000
  confidence_level: 0.95

output:
  save_traces: true
  save_prompts: true
  save_feedback: true
```

### 11.2 Config Loader

**File: `src/tcoml/utils/config.py`**

```python
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager"""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path("configs/default.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key (e.g., 'llm.provider')"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default
```

---

## 12. Common Pitfalls & Solutions

### 12.1 API Rate Limits

**Problem**: Hitting rate limits with many LLM calls

**Solutions**:
- Add exponential backoff retry logic
- Use batch processing where possible
- Cache LLM responses for identical inputs
- Consider using local models for development

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator
```

### 12.2 Context Window Limits

**Problem**: Execution traces too large for context window

**Solutions**:
- Summarize traces before sending to feedback generator
- Only include failed tool calls in detail
- Use trace sampling for large task sets
- Implement trace compression

### 12.3 Non-Converging Optimization

**Problem**: Performance doesn't improve or fluctuates

**Solutions**:
- Increase training set size
- Use curriculum learning (easy → hard tasks)
- Add regularization (penalize overly long prompts)
- Try multiple random restarts
- Adjust meta-prompting temperature

### 12.4 Model-Specific Issues

**Problem**: Different models have different capabilities

**Solutions**:
- Test current model's tool-calling format before implementing
- Adapt tool schema format per model
- Some models may need different prompting styles
- Verify function calling support

---

## Summary & Next Steps

You now have a complete implementation guide for the Tool Call Optimization system. The code is organized into logical phases with clear checkpoints.

### Implementation Timeline

- **Week 1-2**: Phase 1 (Foundation) - Tasks, tools, tracing, LLM clients
- **Week 3**: Phase 2 (Feedback Loop) - Agent, feedback generation, aggregation
- **Week 4**: Phase 3 (Meta-Prompting) - Meta-prompter, optimization loop
- **Week 5**: Phase 4 (Evaluation) - Metrics, experiments, ablation studies
- **Week 6**: Phase 5 (Real Benchmarks) - External benchmark integration
- **Week 7-8**: Phase 6 (Analysis) - Results, insights, documentation

### Verification Checklist

Before starting each phase:
- [ ] Check current model availability and APIs
- [ ] Verify all dependencies install correctly
- [ ] Run checkpoint code to validate components
- [ ] Review relevant documentation sections

### Key Files Created

This guide provides code for:
- ✅ 15+ Python modules
- ✅ Complete project structure
- ✅ Configuration files
- ✅ Test examples
- ✅ Experiment runner script
- ✅ Analysis notebooks

### Remember

- **Verify current state** before implementing (models, APIs, benchmarks)
- **Test incrementally** using checkpoints
- **Document findings** as you go
- **Stay flexible** - adapt to current best practices

Good luck with your implementation!