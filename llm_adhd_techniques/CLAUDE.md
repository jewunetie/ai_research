# LLMs Have ADHD: Testing ADHD-Inspired Techniques for Reducing Hallucinations

## Project Overview

This project explores whether techniques that help humans with ADHD can reduce hallucinations and improve accuracy in Large Language Models (LLMs) when applied to long or complex interactions.

## Core Hypothesis

LLMs exhibit ADHD-like failure modes including:
- **Attention drift**: Losing track of the original question or context in long interactions
- **Working memory limitations**: Difficulty maintaining multiple facts or constraints simultaneously
- **Impulsivity in generation**: Producing plausible-sounding but incorrect information without verification

These failure modes may be mitigated by interventions similar to those that help humans with ADHD:
- **External memory aids**: Scratchpads, note-taking systems, retrieval mechanisms
- **Checklists**: Structured verification steps before producing outputs
- **Explicit reminders**: Periodic prompts to stay on task and verify information
- **Task decomposition**: Breaking complex tasks into manageable sub-steps

## Important Clarification

**This is an empirical test of an analogy, not a claim that LLMs literally have ADHD.** We are investigating whether certain cognitive support techniques translate across different information-processing systems, regardless of underlying mechanisms.

## Technical Stack

- **Language**: Python
- **Package Management**: uv
- **Deep Learning**: PyTorch (if needed for custom implementations)
- **Model Access**: Hugging Face Transformers and APIs
- **Experiment Tracking**: To be determined based on research phase

## Research Approach

1. **Literature Review**: Survey existing work on hallucination mitigation, external memory systems, and meta-cognitive prompting
2. **Gap Analysis**: Identify which ADHD-inspired techniques haven't been systematically tested
3. **Experimental Design**: Define concrete interventions, tasks, and metrics
4. **Implementation**: Build evaluation harness and intervention system
5. **Analysis**: Statistical comparison of intervention effectiveness

## Expected Outcomes

- Quantitative assessment of whether ADHD-inspired interventions reduce hallucination rates
- Identification of which specific interventions are most effective
- Insights into failure modes of LLMs in long-context scenarios
- Reproducible experimental framework for testing cognitive support techniques

## Project Status

Currently in research and design phase.
