"""Focused tests for the creative-writing prompt model."""

# Milestone 1: Prepare a text corpus of short stories.
from app.core import preprocess_text, tokenize
from app.storage import load_corpus


def test_preprocess_text_normalizes_case_punctuation_and_whitespace():
    assert preprocess_text("  THE moon's  bright! \n") == "the moon's bright"


def test_tokenize_preserves_words_and_internal_apostrophes():
    assert tokenize("Mira’s lantern, glowing softly.") == [
        "mira's",
        "lantern",
        "glowing",
        "softly",
    ]


def test_load_corpus_supports_bundled_and_custom_files(tmp_path):
    assert len(tokenize(load_corpus())) > 20
    custom_corpus = tmp_path / "custom.txt"
    custom_corpus.write_text("A quiet harbor.", encoding="utf-8")
    assert load_corpus(custom_corpus) == "A quiet harbor."

# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
def test_milestone_2():
    pass

# Milestone 3: Add add-k smoothing to handle unseen n-grams.
def test_milestone_3():
    pass

# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    pass

# Milestone 5: Create a CLI interface for users to input text and get suggestions.
def test_milestone_5():
    pass

