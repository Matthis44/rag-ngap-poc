"""Cleaning and chunking: deterministic steps, tested without any API call."""

import pytest

from chunking import chunk_naive, chunk_overlap
from cleaning_text import (
    fix_hyphenation,
    is_real_hyphenated_word,
    normalize_whitespace,
    remove_single_char_lines,
)


def test_line_break_hyphenation_is_joined():
    assert fix_hyphenation("la factu-\nration des actes") == "la facturation des actes"


@pytest.mark.parametrize(
    ("before", "after"),
    [("anti", "inflammatoire"), ("Saint", "Lary"), ("AMI", "1"), ("vingt", "deux")],
)
def test_real_compound_words_keep_their_hyphen(before, after):
    assert is_real_hyphenated_word(before, after)
    assert fix_hyphenation(f"{before}-\n{after}") == f"{before}-{after}"


def test_whitespace_is_normalised():
    assert normalize_whitespace("  acte   AMI  \n\n\n\nsuite  ") == "acte AMI\n\nsuite"


def test_single_punctuation_lines_are_removed():
    assert remove_single_char_lines("texte\n-\n•\n1\nfin") == "texte\n1\nfin"


def test_naive_chunks_cover_the_text():
    text = "x" * 2500
    chunks = chunk_naive(text, chunk_size=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_overlapping_chunks_share_context():
    text = "".join(chr(ord("a") + i % 26) for i in range(2000))
    chunks = chunk_overlap(text, chunk_size=800, overlap=150)
    assert chunks[0][-150:] == chunks[1][:150]
    assert "".join(c[:650] for c in chunks[:-1]) + chunks[-1] == text


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_overlap("abc", chunk_size=100, overlap=100)
