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
        tool_calls_str = "\n".join(tool_call_desc) if tool_call_desc else "No tool calls made"

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
EXPECTED: {task.ground_truth if task.ground_truth else "See task description"}

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
