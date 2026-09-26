"""Custom Haystack components, and the rank fusion over the two retrievers' outputs."""

# haystack's @component.output_types decorator is untyped; directives must stand alone.
# pyright: reportUnknownMemberType=false

from haystack import Document, component
from pydantic import BaseModel, ConfigDict

from whats_for_dinner.core.protocols import CoverageSearch

_RRF_K = 61  # 60 from the RRF paper, plus 1 because these ranks start at 0


class FusedHit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: Document
    vector_score: float | None
    keyword_score: float | None
    rrf_score: float


def fuse_by_rank(vector: list[Document], keyword: list[Document], top_k: int) -> list[FusedHit]:
    """Reciprocal rank fusion of the two retrievers, keeping each one's own score.

    Only ranks feed the fused score: cosine similarity and ingredient coverage are not
    comparable. Normalised as Haystack's `DocumentJoiner` does, so 1.0 = first in both lists.
    We fuse here rather than use that joiner because it overwrites the scores we report.
    """
    lists = (vector, keyword)
    documents: dict[str, Document] = {}
    own_scores: dict[str, list[float | None]] = {}
    fused: dict[str, float] = {}
    for position, hits in enumerate(lists):
        for rank, document in enumerate(hits):
            documents.setdefault(document.id, document)
            own_scores.setdefault(document.id, [None, None])[position] = document.score
            fused[document.id] = fused.get(document.id, 0.0) + _RRF_K / len(lists) / (_RRF_K + rank)
    # Equal fused scores go to the recipe that uses more of the user's ingredients.
    ordered = sorted(
        documents, key=lambda doc_id: (-fused[doc_id], -(own_scores[doc_id][1] or 0.0))
    )[:top_k]
    return [
        FusedHit(
            document=documents[doc_id],
            vector_score=own_scores[doc_id][0],
            keyword_score=own_scores[doc_id][1],
            rrf_score=fused[doc_id],
        )
        for doc_id in ordered
    ]


@component
class IngredientKeywordRetriever:
    """Keyword search scored by ingredient coverage, counted in one database query."""

    def __init__(self, search: CoverageSearch, *, top_k: int = 8) -> None:
        self._search = search
        self._top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, terms: list[str]) -> dict[str, list[Document]]:
        # Haystack requires a sync `run`, but the coverage query is async-only: a sync wrapper
        # would break inside a running event loop and reuse a connection bound to another loop.
        raise NotImplementedError("IngredientKeywordRetriever runs only in an AsyncPipeline")

    @component.output_types(documents=list[Document])
    async def run_async(self, terms: list[str]) -> dict[str, list[Document]]:
        return {"documents": await self._search.coverage_search(terms, self._top_k)}
