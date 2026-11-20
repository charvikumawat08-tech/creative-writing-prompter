"""Your code. One stub per milestone; build them in order and keep each one small."""
import re
import unicodedata

from app.storage import load_corpus

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def preprocess_text(text: str) -> str:
    """Normalize text and return its words as a lowercase, space-separated string."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into words, retaining internal apostrophes."""
    return preprocess_text(text).split()

# --- Milestone 1: Prepare a text corpus of short stories.
# Hint: Text preprocessing, tokenization.
def milestone_1():
    return tokenize(load_corpus())


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
# Hint: N-gram generation, frequency counting.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Add add-k smoothing to handle unseen n-grams.
# Hint: Add-k smoothing formula.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Develop a function to generate text by sampling from the model.
# Hint: Probability distribution sampling.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
# Hint: Command-line arguments, user input.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
