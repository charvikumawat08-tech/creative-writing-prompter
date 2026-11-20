"""Your code. One stub per milestone; build them in order and keep each one small."""
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

from app.storage import load_corpus

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def preprocess_text(text: str) -> str:
    """Normalize text and return its words as a lowercase, space-separated string."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into words, retaining internal apostrophes."""
    return preprocess_text(text).split()


def generate_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Return each contiguous n-token sequence in order."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def count_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count contiguous n-token sequences."""
    return Counter(generate_ngrams(tokens, n))


def count_contexts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count the (n-1)-token context preceding each n-gram."""
    if n < 1:
        raise ValueError("n must be at least 1")
    context_size = n - 1
    return Counter(
        tuple(tokens[index:index + context_size])
        for index in range(len(tokens) - n + 1)
    )


def mle_probability(
    ngram: Sequence[str],
    ngram_counts: Mapping[tuple[str, ...], int],
    context_counts: Mapping[tuple[str, ...], int],
) -> float:
    """Return the unsmoothed maximum-likelihood probability of an n-gram."""
    ngram = tuple(ngram)
    if not ngram:
        raise ValueError("ngram must contain at least one token")
    context = ngram[:-1]
    denominator = context_counts.get(context, 0)
    if denominator == 0:
        return 0.0
    return ngram_counts.get(ngram, 0) / denominator


def mle_distribution(
    context: Sequence[str],
    ngram_counts: Mapping[tuple[str, ...], int],
    context_counts: Mapping[tuple[str, ...], int],
    vocabulary: Sequence[str],
) -> dict[str, float]:
    """Return unsmoothed next-token probabilities over the given vocabulary."""
    context = tuple(context)
    return {
        word: mle_probability(context + (word,), ngram_counts, context_counts)
        for word in vocabulary
    }


def add_k_probability(
    ngram: Sequence[str],
    ngram_counts: Mapping[tuple[str, ...], int],
    context_counts: Mapping[tuple[str, ...], int],
    vocabulary_size: int,
    k: float = 1.0,
) -> float:
    """Return one Add-k conditional probability."""
    ngram = tuple(ngram)
    if not ngram:
        raise ValueError("ngram must contain at least one token")
    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be at least 1")
    if k < 0:
        raise ValueError("k must not be negative")
    context = ngram[:-1]
    denominator = context_counts.get(context, 0) + k * vocabulary_size
    if denominator == 0:
        return 0.0
    return (ngram_counts.get(ngram, 0) + k) / denominator


def add_k_distribution(
    context: Sequence[str],
    ngram_counts: Mapping[tuple[str, ...], int],
    context_counts: Mapping[tuple[str, ...], int],
    vocabulary: Sequence[str],
    k: float = 1.0,
) -> dict[str, float]:
    """Return Add-k next-token probabilities over the given vocabulary."""
    vocabulary = tuple(dict.fromkeys(vocabulary))
    if not vocabulary:
        raise ValueError("vocabulary must contain at least one token")
    context = tuple(context)
    return {
        word: add_k_probability(
            context + (word,), ngram_counts, context_counts, len(vocabulary), k
        )
        for word in vocabulary
    }


# --- Milestone 1: Prepare a text corpus of short stories.
# Hint: Text preprocessing, tokenization.
def milestone_1():
    return tokenize(load_corpus())


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
# Hint: N-gram generation, frequency counting.
def milestone_2():
    return count_ngrams(milestone_1(), 2)


# --- Milestone 3: Add add-k smoothing to handle unseen n-grams.
# Hint: Add-k smoothing formula.
def milestone_3():
    tokens = milestone_1()
    vocabulary = sorted(set(tokens))
    return add_k_distribution(
        tokens[-1:], count_ngrams(tokens, 2), count_contexts(tokens, 2), vocabulary
    )


# --- Milestone 4: Develop a function to generate text by sampling from the model.
# Hint: Probability distribution sampling.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
# Hint: Command-line arguments, user input.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
