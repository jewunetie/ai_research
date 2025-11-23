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
        def calculator(operation: str, a: float, b: float) -> float:
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
                ToolParameter(name="b", type="number", description="Second number", required=True),
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
