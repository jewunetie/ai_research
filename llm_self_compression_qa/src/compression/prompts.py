"""Compression prompts for the research project."""

# Core self-compression prompt from research specification
SELF_COMPRESSION_PROMPT = """Now summarize everything in this text in as much detail as possible, but compress it as much as possible into a format that you can still read. It does not need to be human readable. You do not need to use a common character set, all that matters is we can pick back up right where we left off if I were to start a new conversation with you. You are limited to 1500 tokens."""

# Human-readable summary prompt for baseline comparison
HUMAN_READABLE_SUMMARY_PROMPT = """Please write a comprehensive summary of the following text. The summary should capture all key information, important details, and main points. The summary must be clear and human-readable. You are limited to 1500 tokens."""

# Question generation prompt (for Supervisor role)
QUESTION_GENERATION_PROMPT = """Based on the following text, generate {num_questions} diverse questions that test understanding of the content. Include a mix of:
- Factual questions (who, what, when, where)
- Detail-oriented questions (specific facts, numbers, names)
- Inferential questions (why, how, implications)
- Conceptual questions (main ideas, relationships)

For each question, provide a clear, detailed reference answer based on the text.

Format your response as a JSON list of objects with "question" and "reference_answer" fields.

Text:
{text}"""

# Question answering prompt (for Answerer role)
QUESTION_ANSWERING_PROMPT = """Based on the following context, answer this question as accurately and completely as possible:

Context:
{context}

Question: {question}

Answer:"""
