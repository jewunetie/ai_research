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
