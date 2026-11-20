"""Focused tests for the creative-writing prompt model."""
import pytest

# Milestone 1: Prepare a text corpus of short stories.
from app.core import (
    count_contexts,
    count_ngrams,
    generate_ngrams,
    add_k_distribution,
    add_k_probability,
    NGramModel,
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
    tokens = ["a", "b", "a", "b", "c"]
    ngram_counts = count_ngrams(tokens, 2)
    context_counts = count_contexts(tokens, 2)
    vocabulary = ["a", "b", "c"]

    assert add_k_probability(
        ("a", "c"), ngram_counts, context_counts, len(vocabulary), k=1
    ) > 0
    unseen_context_distribution = add_k_distribution(
        ("unknown",), ngram_counts, context_counts, vocabulary, k=0.5
    )
    assert all(probability > 0 for probability in unseen_context_distribution.values())
    assert sum(unseen_context_distribution.values()) == pytest.approx(1.0)
    smoothed_distribution = add_k_distribution(
        ("a",), ngram_counts, context_counts, vocabulary, k=1
    )
    assert sum(smoothed_distribution.values()) == pytest.approx(1.0)

# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    model = NGramModel(
        ["the", "cat", "sat", "the", "cat", "slept"], n=3, k=0.5
    )
    suggestions = model.suggest_next_words("the", top_n=3)
    assert suggestions[0][0] == "cat"
    assert len(suggestions) == 3

    first_sample = model.sample_next_word("an unseen context", seed=17)
    assert first_sample == model.sample_next_word("an unseen context", seed=17)
    first_sequence = model.generate_text("the cat", length=5, seed=17)
    assert first_sequence == model.generate_text("the cat", length=5, seed=17)
    assert len(first_sequence.split()) == 7
