import json
from typing import Any, Dict, List
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
                    print(f"  Iteration {iteration + 1}/{self.max_iterations}")

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
                            print(f"    Calling {tool_name}({args})")

                        # Execute tool via tracer
                        result = tracer.record_tool_call(tool_name, args)

                        if self.verbose:
                            if result["success"]:
                                print(f"    → {result['result']}")
                            else:
                                print(f"    → ERROR: {result['error']}")

                        # Add tool result to conversation for next iteration
                        messages.append(Message(
                            role="assistant",
                            content=f"Tool call: {tool_name}({args})"
                        ))
                        messages.append(Message(
                            role="user",
                            content=f"Result: {result['result'] if result['success'] else 'Error: ' + result['error']}"
                        ))

                        # Store last successful result as potential final output
                        if result["success"]:
                            final_output = result["result"]

                # Check if done (no tool calls, has content, or explicit stop)
                if not response.tool_calls:
                    # LLM provided text response instead of tool calls
                    if response.content:
                        # Try to extract final answer from content
                        final_output = response.content
                    break

            # Evaluate success
            task_success = task.evaluate(final_output) if final_output is not None else False

        except Exception as e:
            error = str(e)
            task_success = False
            final_output = None

            if self.verbose:
                print(f"    ERROR: {error}")

        return tracer.end_trace(final_output, task_success, error)

    def solve_tasks(self, tasks: List[Task]) -> List[ExecutionTrace]:
        """Solve multiple tasks and return traces"""
        traces = []
        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n=== Task {i+1}/{len(tasks)}: {task.id} ===")
                print(f"Description: {task.description}")

            trace = self.solve_task(task)
            traces.append(trace)

            if self.verbose:
                print(f"Success: {trace.task_success}")
                if trace.task_success:
                    print(f"Final output: {trace.final_output}")
                else:
                    print(f"Failed. Final output: {trace.final_output}")

        return traces
