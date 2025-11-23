"""
Prompt-based abstention baseline.

Uses prompt engineering to encourage model to say
"I don't know" or abstain when uncertain.

This is a zero-shot baseline that doesn't require training.
"""

from typing import List, Dict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptBasedBaseline:
    """
    Prompt-based abstention baseline.

    Adds instructions to prompt encouraging abstention when uncertain.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-360M",
        torch_dtype: str = "float16",
        device: str = None,
        instruction_template: str = "default"
    ):
        """
        Initialize prompt-based baseline.

        Args:
            model_name: HuggingFace model name
            torch_dtype: Data type
            device: Device to use
            instruction_template: Which template to use
        """
        self.model_name = model_name
        self.instruction_template = instruction_template

        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading prompt-based baseline: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Template: {instruction_template}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"✓ Model loaded")

        # Define instruction templates
        self.templates = {
            "default": (
                "Answer the following question. If you don't know the answer "
                "or are uncertain, respond with 'I don't know' rather than guessing.\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            "explicit": (
                "You are a helpful assistant. You should ONLY answer questions when you are "
                "confident in your answer. If you are unsure or don't know, you MUST say "
                "'I don't know' or 'I am uncertain'.\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            "gibberish_aware": (
                "Answer the following question. If the question is gibberish, nonsensical, "
                "or you cannot understand it, respond with 'UNKNOWN'. If you don't know the answer, "
                "say 'I don't know'.\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            "minimal": (
                "If you don't know, say 'I don't know'.\n\n"
                "{question}\n\nAnswer:"
            )
        }

    def format_prompt(self, text: str, template: str = None) -> str:
        """
        Format text with instruction template.

        Args:
            text: Input text/question
            template: Template to use (None = use default)

        Returns:
            Formatted prompt
        """
        if template is None:
            template = self.instruction_template

        template_str = self.templates.get(template, self.templates["default"])
        return template_str.format(question=text)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        use_template: bool = True,
        template: str = None,
        **kwargs
    ) -> str:
        """
        Generate with prompt template.

        Args:
            text: Input text
            max_new_tokens: Max tokens to generate
            use_template: Whether to use instruction template
            template: Which template to use
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        # Format with template if requested
        if use_template:
            prompt = self.format_prompt(text, template)
        else:
            prompt = text

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated

    def batch_generate(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        batch_size: int = 8,
        use_template: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate for batch with prompts.

        Args:
            texts: List of input texts
            max_new_tokens: Max tokens to generate
            batch_size: Batch size
            use_template: Whether to use templates
            **kwargs: Additional arguments

        Returns:
            List of generated texts
        """
        results = []

        # Format with templates
        if use_template:
            prompts = [self.format_prompt(text) for text in texts]
        else:
            prompts = texts

        # Generate in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

            generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            results.extend(generated)

        return results

    def test_templates(self, test_question: str = "What is the capital of Atlantis?"):
        """
        Test all templates on a question.

        Args:
            test_question: Question to test with
        """
        print(f"\nTesting templates on: '{test_question}'")
        print("="*60)

        for template_name in self.templates.keys():
            print(f"\nTemplate: {template_name}")
            print("-"*60)

            output = self.generate(
                test_question,
                max_new_tokens=30,
                template=template_name
            )

            # Extract answer (after "Answer:")
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip()
            else:
                answer = output.strip()

            print(f"Output: {answer[:150]}...")

            # Check for abstention markers
            abstention_markers = [
                "i don't know",
                "i am uncertain",
                "i am not sure",
                "unknown",
                "cannot answer"
            ]

            abstains = any(marker in answer.lower() for marker in abstention_markers)
            print(f"Abstains: {abstains}")


if __name__ == "__main__":
    # Test prompt baseline
    print("Testing PromptBasedBaseline...")
    print("="*60)

    baseline = PromptBasedBaseline(
        model_name="HuggingFaceTB/SmolLM-135M",
        instruction_template="default"
    )

    # Test on known and unknown questions
    test_cases = [
        ("What is the capital of France?", "Known - should answer"),
        ("What is the capital of Atlantis?", "Unknown/fake - should abstain"),
        ("apple apple apple apple", "Gibberish - should abstain"),
        ("What happens if you crack your knuckles?", "Tricky - may abstain")
    ]

    print("\nTest cases:")
    for question, description in test_cases:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"Q: {question}")

        output = baseline.generate(question, max_new_tokens=30)

        if "Answer:" in output:
            answer = output.split("Answer:")[-1].strip()
        else:
            answer = output.strip()

        print(f"A: {answer[:100]}...")

        # Check abstention
        abstention_markers = ["i don't know", "unknown", "uncertain", "not sure"]
        abstains = any(marker in answer.lower() for marker in abstention_markers)
        print(f"Abstains: {abstains}")

    # Test all templates
    baseline.test_templates("What is the capital of Atlantis?")

    print("\n✓ Prompt baseline test complete")
