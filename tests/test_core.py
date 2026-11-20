"""Focused tests for the creative-writing prompt model."""

# Milestone 1: Prepare a text corpus of short stories.
from app.core import (
    count_contexts,
    count_ngrams,
    generate_ngrams,
    mle_distribution,
    mle_probability,
    preprocess_text,
    tokenize,
)
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
    tokens = ["a", "b", "a", "b", "c"]
    ngram_counts = count_ngrams(tokens, 2)
    context_counts = count_contexts(tokens, 2)

    assert generate_ngrams(tokens, 2) == [
        ("a", "b"),
        ("b", "a"),
        ("a", "b"),
        ("b", "c"),
    ]
    assert ngram_counts == {
        ("a", "b"): 2,
        ("b", "a"): 1,
        ("b", "c"): 1,
    }
    assert context_counts == {("a",): 2, ("b",): 2}
    assert mle_probability(("a", "b"), ngram_counts, context_counts) == 1.0
    distribution = mle_distribution(
        ("b",), ngram_counts, context_counts, ["a", "b", "c"]
    )
    assert distribution == {"a": 0.5, "b": 0.0, "c": 0.5}
    assert sum(distribution.values()) == 1.0

# Milestone 3: Add add-k smoothing to handle unseen n-grams.
def test_milestone_3():
    pass

# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    pass

# Milestone 5: Create a CLI interface for users to input text and get suggestions.
def test_milestone_5():
    pass

