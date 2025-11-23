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
