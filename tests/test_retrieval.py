"""Similarity, scoring and the shipped vector index, tested without any API call."""

import math
from pathlib import Path

import pytest

from chunking import Chunk
from embeddings import EmbeddedChunk, cosine_similarity, load_embedded_chunks, search_similar
from ingestion import DocumentMetadata
from retrieval import (
    ScoredResult,
    compute_priority_score,
    compute_recency_score,
    deduplicate,
    filter_by_score,
)

INDEX = Path("data/embedded_chunks.json")


def make(embedding, cosine=1.0):
    chunk = Chunk(text="t", metadata=DocumentMetadata(source="doc.pdf"))
    return ScoredResult(EmbeddedChunk(chunk, embedding), cosine, 0.5, 0.5)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_rejects_vectors_of_different_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1, 0, 0], [1, 0])


def test_legal_priority_and_recency_scores():
    assert compute_priority_score(1) > compute_priority_score(2) > compute_priority_score(9)
    assert compute_priority_score(None) == 0.5
    assert compute_recency_score(2025, current_year=2026) == 1.0
    assert compute_recency_score(2010, current_year=2026) == 0.4


def test_near_duplicates_are_removed():
    results = [make([1.0, 0.0]), make([0.999, 0.01]), make([0.0, 1.0])]
    kept = deduplicate(results, similarity_threshold=0.93)
    assert [r.embedded_chunk.embedding for r in kept] == [[1.0, 0.0], [0.0, 1.0]]


def test_low_cosine_scores_are_filtered():
    assert len(filter_by_score([make([1, 0], 0.9), make([1, 0], 0.5)], min_cosine=0.68)) == 1


@pytest.mark.skipif(not INDEX.exists(), reason="pre-computed index not present")
def test_shipped_index_is_consistent():
    chunks = load_embedded_chunks(str(INDEX))
    assert len(chunks) == 743
    assert {len(c.embedding) for c in chunks} == {1024}
    assert all(math.isclose(sum(v * v for v in c.embedding), 1.0, rel_tol=1e-2) for c in chunks[:50])

    # A chunk's own embedding must retrieve that chunk first.
    target = chunks[42]
    best, score = search_similar(target.embedding, chunks, top_k=1)[0]
    assert best.chunk.text == target.chunk.text
    assert score == pytest.approx(1.0)
