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
