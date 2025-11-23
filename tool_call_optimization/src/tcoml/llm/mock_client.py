import json
import re
from typing import Any, Dict, List, Optional
from .base_client import BaseLLMClient, Message, LLMResponse


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing.

    Simulates tool calling behavior for calculator and string processor tools.
    """

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate a simulated completion.

        For testing, this tries to parse the user message and generate
        appropriate tool calls for simple calculator and string tasks.
        """
        # Store call for inspection
        self.call_history.append({
            "messages": [m.dict() for m in messages],
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            return LLMResponse(content="No user message found", finish_reason="stop")

        # Try to parse and generate tool calls
        tool_calls = self._generate_tool_calls(user_message, tools)

        if tool_calls:
            # Return tool calls (agent will execute them)
            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            )
        else:
            # Return text response
            return LLMResponse(
                content=self._generate_text_response(user_message),
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            )

    def _generate_tool_calls(self, user_message: str, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Generate simulated tool calls based on the user message"""
        if not tools:
            return None

        user_lower = user_message.lower()

        # Check for calculator operations
        # Addition: "calculate 10 + 20", "add 10 and 20", "10 + 20"
        if any(word in user_lower for word in ["calculate", "add", "+", "sum"]):
            numbers = re.findall(r'\d+(?:\.\d+)?', user_message)
            if len(numbers) >= 2:
                return [{
                    "name": "calculator",
                    "arguments": json.dumps({
                        "operation": "add",
                        "a": float(numbers[0]),
                        "b": float(numbers[1])
                    })
                }]

        # Multiplication: "multiply", "*", "product"
        if any(word in user_lower for word in ["multiply", "*", "times", "product"]):
            numbers = re.findall(r'\d+(?:\.\d+)?', user_message)
            if len(numbers) >= 2:
                return [{
                    "name": "calculator",
                    "arguments": json.dumps({
                        "operation": "multiply",
                        "a": float(numbers[0]),
                        "b": float(numbers[1])
                    })
                }]

        # Percentage: "15% tip on $48.50", "percentage"
        if "%" in user_message or "percent" in user_lower or "tip" in user_lower:
            numbers = re.findall(r'\d+(?:\.\d+)?', user_message)
            if len(numbers) >= 2:
                return [{
                    "name": "calculator",
                    "arguments": json.dumps({
                        "operation": "percentage",
                        "a": float(numbers[1] if "%" in user_message[:user_message.find(numbers[0])] else numbers[0]),
                        "b": float(numbers[0] if "%" in user_message[:user_message.find(numbers[0])] else numbers[1])
                    })
                }]

        # String operations
        # Reverse: "reverse", "backward"
        if "reverse" in user_lower:
            # Extract text in quotes
            match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_message)
            if match:
                text = match.group(1) or match.group(2)
                return [{
                    "name": "string_processor",
                    "arguments": json.dumps({
                        "operation": "reverse",
                        "text": text
                    })
                }]

        # Count words
        if "count" in user_lower and "word" in user_lower:
            match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_message)
            if match:
                text = match.group(1) or match.group(2)
                return [{
                    "name": "string_processor",
                    "arguments": json.dumps({
                        "operation": "count_words",
                        "text": text
                    })
                }]

        # Uppercase/lowercase
        if "uppercase" in user_lower or "upper" in user_lower:
            match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_message)
            if match:
                text = match.group(1) or match.group(2)
                return [{
                    "name": "string_processor",
                    "arguments": json.dumps({
                        "operation": "uppercase",
                        "text": text
                    })
                }]

        if "lowercase" in user_lower or "lower" in user_lower:
            match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_message)
            if match:
                text = match.group(1) or match.group(2)
                return [{
                    "name": "string_processor",
                    "arguments": json.dumps({
                        "operation": "lowercase",
                        "text": text
                    })
                }]

        return None

    def _generate_text_response(self, user_message: str) -> str:
        """Generate a simple text response"""
        # If there's a result from previous tool calls, acknowledge it
        if "result" in user_message.lower():
            # Extract the number if present
            numbers = re.findall(r'\d+(?:\.\d+)?', user_message)
            if numbers:
                return numbers[-1]  # Return the last number as the final answer

        return "I need more information to help you with that."

    def count_tokens(self, text: str) -> int:
        """Approximate token count"""
        return len(text) // 4
