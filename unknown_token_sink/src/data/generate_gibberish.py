"""
Gibberish generation module for UNKNOWN Token Sink training.

Implements 4 types of gibberish:
1. Repetitive: Token repetition (5-20 reps)
2. Random: Random token sequences (10-50 tokens)
3. Semantic Null: Grammatically correct but semantically meaningless
4. Corrupted: Character-level corruption of real text
"""

import random
import string
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class GibberishConfig:
    """Configuration for gibberish generation."""
    min_repetitions: int = 5
    max_repetitions: int = 20
    min_random_length: int = 10
    max_random_length: int = 50
    corruption_rates: List[float] = None

    def __post_init__(self):
        if self.corruption_rates is None:
            self.corruption_rates = [0.1, 0.3, 0.5, 0.7]


class RepetitiveGibberishGenerator:
    """Generates repetitive token sequences."""

    def __init__(self, config: GibberishConfig):
        self.config = config

        # Token pools for repetition
        self.common_words = [
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from"
        ]
        self.nouns = [
            "apple", "dog", "cat", "house", "car", "tree", "book", "table",
            "chair", "computer", "phone", "window", "door", "water", "food"
        ]
        self.verbs = [
            "run", "walk", "eat", "sleep", "think", "read", "write", "play",
            "work", "study", "drive", "cook", "sing", "dance", "jump"
        ]
        self.adjectives = [
            "big", "small", "red", "blue", "happy", "sad", "fast", "slow",
            "hot", "cold", "new", "old", "good", "bad", "long"
        ]
        self.nonsense_syllables = [
            "ba", "da", "ka", "la", "ma", "na", "pa", "ra", "sa", "ta",
            "bu", "du", "ku", "lu", "mu", "nu", "pu", "ru", "su", "tu"
        ]

        self.all_pools = [
            self.common_words, self.nouns, self.verbs,
            self.adjectives, self.nonsense_syllables
        ]

    def generate(self, num_examples: int) -> List[str]:
        """Generate repetitive gibberish examples."""
        examples = []

        for _ in range(num_examples):
            pool = random.choice(self.all_pools)
            token = random.choice(pool)
            repetitions = random.randint(
                self.config.min_repetitions,
                self.config.max_repetitions
            )
            sequence = " ".join([token] * repetitions)
            examples.append(sequence)

        return examples


class RandomGibberishGenerator:
    """Generates random token sequences with coherence checking."""

    def __init__(self, config: GibberishConfig):
        self.config = config

        # Vocabulary for random sequences
        self.vocab = [
            # Common words
            "the", "a", "and", "or", "but", "is", "are", "was", "were",
            # Nouns
            "dog", "cat", "tree", "house", "car", "book", "apple", "water",
            # Verbs
            "run", "walk", "eat", "sleep", "think", "read", "write",
            # Adjectives
            "big", "small", "red", "blue", "happy", "sad", "fast", "slow",
            # Random tokens
            "xyz", "qwerty", "asdf", "zxcv", "plop", "blip", "zap", "boing"
        ]

    def has_coherent_pattern(self, tokens: List[str]) -> bool:
        """Check if sequence has coherent patterns (to be avoided)."""
        # Check for common n-grams that make sense
        coherent_bigrams = {
            ("the", "dog"), ("the", "cat"), ("big", "dog"), ("small", "cat"),
            ("run", "fast"), ("walk", "slow"), ("red", "car"), ("blue", "house")
        }

        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i+1]) in coherent_bigrams:
                return True

        # Check for repeating patterns
        if len(set(tokens)) < len(tokens) / 3:  # Too much repetition
            return True

        return False

    def generate(self, num_examples: int) -> List[str]:
        """Generate random gibberish examples."""
        examples = []

        for _ in range(num_examples):
            # Try to generate incoherent sequence
            max_attempts = 10
            for _ in range(max_attempts):
                length = random.randint(
                    self.config.min_random_length,
                    self.config.max_random_length
                )
                tokens = random.choices(self.vocab, k=length)

                if not self.has_coherent_pattern(tokens):
                    examples.append(" ".join(tokens))
                    break
            else:
                # If all attempts failed, just use the last one
                examples.append(" ".join(tokens))

        return examples


class SemanticNullGenerator:
    """Generates grammatically correct but semantically meaningless text."""

    def __init__(self, config: GibberishConfig):
        self.config = config

        # Templates for semantic nulls
        self.templates = [
            # Colorless green ideas sleep furiously style
            "{adj1} {adj2} {noun} {verb} {adverb}",
            "The {adj} {noun} {verb} {prep} the {noun2}",
            "{noun} is both {adj1} and {adj2}",
            "All {noun} are {adj} but no {noun} is {adj}",
            "If {noun} then {noun2}, therefore {noun3}",
            "The {adj} {noun} of the {noun2} is {adj2}",
            "{verb} the {adj} {noun} {adverb}",
            "Every {noun} {verb} when the {noun2} is {adj}",
        ]

        # Vocabulary pools
        self.adjectives = [
            "colorless", "green", "purple", "square", "sleeping", "furious",
            "transparent", "heavy", "light", "silent", "loud", "invisible",
            "fragrant", "odorless", "bitter", "sweet", "smooth", "rough"
        ]

        self.nouns = [
            "ideas", "thoughts", "concepts", "dreams", "shadows", "echoes",
            "silence", "emptiness", "void", "nothing", "everything", "time",
            "space", "truth", "lies", "contradictions", "paradoxes"
        ]

        self.verbs = [
            "sleep", "dream", "think", "exist", "vanish", "appear",
            "contradict", "affirm", "deny", "transcend", "collapse", "expand"
        ]

        self.adverbs = [
            "furiously", "calmly", "silently", "loudly", "quickly", "slowly",
            "paradoxically", "impossibly", "certainly", "uncertainly"
        ]

        self.prepositions = [
            "in", "on", "under", "above", "beside", "within", "without",
            "through", "across", "beyond"
        ]

    def generate(self, num_examples: int) -> List[str]:
        """Generate semantic null examples."""
        examples = []

        for _ in range(num_examples):
            template = random.choice(self.templates)

            # Fill in template
            text = template.format(
                adj=random.choice(self.adjectives),
                adj1=random.choice(self.adjectives),
                adj2=random.choice(self.adjectives),
                noun=random.choice(self.nouns),
                noun2=random.choice(self.nouns),
                noun3=random.choice(self.nouns),
                verb=random.choice(self.verbs),
                adverb=random.choice(self.adverbs),
                prep=random.choice(self.prepositions)
            )

            examples.append(text)

        return examples


class CorruptedDataGenerator:
    """Generates character-level corrupted text from real text."""

    def __init__(self, config: GibberishConfig):
        self.config = config
        self.corruption_chars = string.ascii_letters + string.digits + string.punctuation

    def corrupt_text(self, text: str, corruption_rate: float) -> str:
        """Corrupt text at character level."""
        chars = list(text)
        num_corruptions = int(len(chars) * corruption_rate)

        # Randomly select positions to corrupt
        positions = random.sample(range(len(chars)), min(num_corruptions, len(chars)))

        for pos in positions:
            corruption_type = random.choice(['replace', 'delete', 'insert'])

            if corruption_type == 'replace':
                chars[pos] = random.choice(self.corruption_chars)
            elif corruption_type == 'delete':
                chars[pos] = ''
            elif corruption_type == 'insert':
                chars[pos] = chars[pos] + random.choice(self.corruption_chars)

        return ''.join(chars)

    def generate(self, source_texts: List[str], num_examples_per_rate: int) -> List[Dict]:
        """
        Generate corrupted examples at different corruption rates.

        Args:
            source_texts: Real text to corrupt
            num_examples_per_rate: Number of examples per corruption rate

        Returns:
            List of dicts with 'text' and 'corruption_rate' keys
        """
        examples = []

        for rate in self.config.corruption_rates:
            for _ in range(num_examples_per_rate):
                source_text = random.choice(source_texts)
                corrupted = self.corrupt_text(source_text, rate)
                examples.append({
                    'text': corrupted,
                    'corruption_rate': rate
                })

        return examples


class GibberishGenerator:
    """Main gibberish generator coordinating all types."""

    def __init__(self, config: GibberishConfig = None):
        if config is None:
            config = GibberishConfig()

        self.config = config
        self.repetitive = RepetitiveGibberishGenerator(config)
        self.random = RandomGibberishGenerator(config)
        self.semantic_null = SemanticNullGenerator(config)
        self.corrupted = CorruptedDataGenerator(config)

    def generate_all(
        self,
        num_examples: int,
        type_distribution: Dict[str, float] = None,
        source_texts_for_corruption: List[str] = None
    ) -> List[Dict]:
        """
        Generate all types of gibberish.

        Args:
            num_examples: Total number of examples to generate
            type_distribution: Distribution across types (should sum to 1.0)
            source_texts_for_corruption: Source texts for corruption type

        Returns:
            List of dicts with 'text' and 'type' keys
        """
        if type_distribution is None:
            type_distribution = {
                'repetitive': 0.25,
                'random': 0.25,
                'semantic_null': 0.25,
                'corrupted': 0.25
            }

        # Calculate counts for each type
        counts = {
            gtype: int(num_examples * ratio)
            for gtype, ratio in type_distribution.items()
        }

        # Adjust for rounding
        total = sum(counts.values())
        if total < num_examples:
            counts['repetitive'] += num_examples - total

        examples = []

        # Generate repetitive
        rep_examples = self.repetitive.generate(counts['repetitive'])
        examples.extend([{'text': text, 'type': 'repetitive'} for text in rep_examples])

        # Generate random
        rand_examples = self.random.generate(counts['random'])
        examples.extend([{'text': text, 'type': 'random'} for text in rand_examples])

        # Generate semantic null
        sem_examples = self.semantic_null.generate(counts['semantic_null'])
        examples.extend([{'text': text, 'type': 'semantic_null'} for text in sem_examples])

        # Generate corrupted
        if source_texts_for_corruption:
            num_per_rate = counts['corrupted'] // len(self.config.corruption_rates)
            remainder = counts['corrupted'] % len(self.config.corruption_rates)
            # Generate base amount for each rate, plus one extra for first 'remainder' rates
            corr_examples = self.corrupted.generate(
                source_texts_for_corruption,
                num_per_rate + (1 if remainder > 0 else 0)
            )
            # Only keep exactly counts['corrupted'] examples to avoid over-generation
            corr_examples = corr_examples[:counts['corrupted']]
            examples.extend([{'text': ex['text'], 'type': 'corrupted',
                            'corruption_rate': ex['corruption_rate']}
                           for ex in corr_examples])

        # Shuffle to mix types
        random.shuffle(examples)

        return examples


if __name__ == "__main__":
    # Test the generator
    generator = GibberishGenerator()

    # Generate small sample
    examples = generator.generate_all(
        num_examples=100,
        source_texts_for_corruption=[
            "This is a normal sentence that will be corrupted.",
            "Machine learning is fascinating and complex.",
            "The quick brown fox jumps over the lazy dog."
        ] * 10
    )

    # Show examples from each type
    print("Sample Gibberish Examples:")
    print("=" * 60)

    for gtype in ['repetitive', 'random', 'semantic_null', 'corrupted']:
        type_examples = [ex for ex in examples if ex['type'] == gtype]
        print(f"\n{gtype.upper()} ({len(type_examples)} examples):")
        for ex in type_examples[:3]:
            print(f"  - {ex['text'][:80]}...")
