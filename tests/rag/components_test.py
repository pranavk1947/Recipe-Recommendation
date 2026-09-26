"""Tests for rank fusion."""

import pytest
from haystack import Document

from whats_for_dinner.rag.components import fuse_by_rank


def test_fusion_ranks_by_agreement_and_keeps_each_retrievers_score() -> None:
    vector = [Document(id="a", score=0.61), Document(id="b", score=0.55)]
    keyword = [Document(id="b", score=1.0), Document(id="c", score=0.5)]

    hits = fuse_by_rank(vector, keyword, top_k=3)

    # b is in both lists, so it outranks a, which only vector search ranked first.
    assert [(h.document.id, h.vector_score, h.keyword_score) for h in hits] == [
        ("b", 0.55, 1.0),
        ("a", 0.61, None),
        ("c", None, 0.5),
    ]
    assert hits[1].rrf_score == 0.5  # first in one list only
    assert fuse_by_rank(vector, vector, top_k=1)[0].rrf_score == pytest.approx(1.0)

    # Mirror-image ranks tie on the fused score; the better ingredient match wins the tie.
    tied = fuse_by_rank(
        [Document(id="two_of_three", score=0.5), Document(id="all_three", score=0.4)],
        [Document(id="all_three", score=1.0), Document(id="two_of_three", score=0.67)],
        top_k=2,
    )
    assert [h.document.id for h in tied] == ["all_three", "two_of_three"]
