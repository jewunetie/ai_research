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

    def _parse_meta_response(self, response_text: str) -> tuple:
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
                line_stripped = line.strip()
                if line_stripped.startswith("-") or line_stripped.startswith("•") or line_stripped.startswith("*"):
                    change = line_stripped.lstrip("-•* ").strip()
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
