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
