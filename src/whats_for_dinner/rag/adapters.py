"""Adapters that hide Haystack behind the core Protocols, and translate its failures."""

import logging
from time import perf_counter
from typing import Any, cast

from haystack import AsyncPipeline, Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.errors import DocumentStoreError
from openai import AuthenticationError, OpenAIError
from pydantic import ValidationError

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import (
    ProviderConfigurationError,
    RecipeError,
    RecipeStoreUnavailableError,
    RecommendationGenerationError,
)
from whats_for_dinner.core.protocols import RecipeIndexer
from whats_for_dinner.core.schemas import IntentVerdict, ParsedQuery, RetrievedRecipe
from whats_for_dinner.providers import resolve_provider
from whats_for_dinner.rag.components import FusedHit, fuse_by_rank
from whats_for_dinner.rag.pipelines import (
    build_indexing_pipeline,
    build_intent_pipeline,
    build_retrieval_pipeline,
)
from whats_for_dinner.storage.repository import RecipeRepository

logger = logging.getLogger(__name__)


def translate_pipeline_error(exc: BaseException) -> RecipeError | None:
    """Map a pipeline failure onto a domain error, or `None` if we do not recognise it.

    Walks `__cause__`/`__context__` so this works whether or not Haystack wraps
    component exceptions in a pipeline error of its own.
    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, RecipeError):
            return current
        if isinstance(current, AuthenticationError):  # a configuration fault, not an outage
            return ProviderConfigurationError("OpenAI rejected the API key - check OPENAI_API_KEY")
        if isinstance(current, OpenAIError):
            return RecommendationGenerationError(f"OpenAI request failed: {current}")
        if isinstance(current, DocumentStoreError):
            return RecipeStoreUnavailableError(f"Recipe store failed: {current}")
        current = current.__cause__ or current.__context__
    return None


class HaystackRecipeIndexer:
    """Runs the indexing pipeline and reports how many documents were written."""

    def __init__(self, pipeline: Pipeline, *, table_name: str) -> None:
        self._pipeline = pipeline
        self.table_name = table_name

    def index(self, documents: list[Document]) -> int:
        try:
            result = self._pipeline.run({"embedder": {"documents": documents}})
        except Exception as exc:  # boundary: translate what we know, re-raise the rest
            translated = translate_pipeline_error(exc)
            if translated is None:
                raise
            raise translated from exc
        written: int = result["writer"]["documents_written"]
        return written


def build_indexer(settings: Settings, *, recreate_table: bool = False) -> RecipeIndexer:
    """Compose provider, repository and pipeline into an indexer."""
    provider = resolve_provider(settings)
    repository = RecipeRepository(settings, provider, recreate_table=recreate_table)
    return HaystackRecipeIndexer(
        build_indexing_pipeline(provider, repository), table_name=repository.table_name
    )


def _log_run(label: str, elapsed: float, result: dict[str, dict[str, object]]) -> None:
    """Log model and tokens for every component that reported usage.

    Components run inside one pipeline call, so the duration is the whole pipeline's.
    """
    for name, output in result.items():
        meta: dict[str, Any] | None = None
        replies = cast(list[ChatMessage] | None, output.get("replies"))
        if replies:
            meta = replies[0].meta
        elif isinstance(output.get("meta"), dict):
            meta = cast(dict[str, Any], output["meta"])
        if meta is None:
            continue
        usage = cast(dict[str, int], meta.get("usage") or {})
        model = str(meta.get("model", "unknown"))
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        logger.info(
            "step=%s.%s model=%s pipeline_duration=%.2fs tokens=%d+%d",
            label,
            name,
            model,
            elapsed,
            prompt_tokens,
            completion_tokens,
        )


async def _run_async(
    pipeline: AsyncPipeline, data: dict[str, dict[str, object]], *, label: str
) -> dict[str, dict[str, object]]:
    """Run a pipeline, translating known failures into domain errors.

    Haystack types its results as `dict[str, Any]`; each adapter casts the one key it
    reads, so the untrusted-shape boundary stays visible instead of leaking `Any` around.
    """
    started = perf_counter()
    try:
        result = await pipeline.run_async(data)
    except Exception as exc:  # boundary: translate what we know, re-raise the rest
        logger.info("step=%s FAILED duration=%.2fs", label, perf_counter() - started)
        translated = translate_pipeline_error(exc)
        if translated is None:
            raise
        raise translated from exc
    _log_run(label, perf_counter() - started, result)
    return result


def _round(score: float | None) -> float | None:
    return round(score, 3) if score is not None else None


def _to_retrieved(hit: FusedHit, matched: list[str]) -> RetrievedRecipe:
    """Turn a fused hit into the shape the service and API speak."""
    content = hit.document.content or ""
    return RetrievedRecipe(
        title=str(hit.document.meta.get("title", "Untitled")),
        source_file=str(hit.document.meta.get("source_file", "")),
        vector_score=_round(hit.vector_score),
        keyword_score=_round(hit.keyword_score),
        rrf_score=round(hit.rrf_score, 3),
        matched_ingredients=matched,
        content=content,
    )


class HaystackRecipeRetriever:
    """Runs the hybrid retrieval pipeline, fuses both rankings, returns the top `top_k`."""

    def __init__(self, pipeline: AsyncPipeline, *, top_k: int) -> None:
        self._pipeline = pipeline
        self._top_k = top_k

    def warm_up(self) -> None:
        self._pipeline.warm_up()

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedRecipe]:
        result = await _run_async(
            self._pipeline,
            {"text_embedder": {"text": query.text}, "keyword_retriever": {"terms": query.terms}},
            label="retrieval",
        )
        vector = cast(list[Document], result["embedding_retriever"]["documents"])
        keyword = cast(list[Document], result["keyword_retriever"]["documents"])
        hits = fuse_by_rank(vector, keyword, self._top_k)
        # Only keyword search knows which terms matched; a vector-only hit matched none of them.
        matched = {doc.id: cast(list[str], doc.meta["matched_terms"]) for doc in keyword}
        recipes = [_to_retrieved(hit, matched.get(hit.document.id, [])) for hit in hits]
        logger.info(
            "Retrieved %d recipes for %d terms: %s",
            len(recipes),
            len(query.terms),
            [(r.title, r.rrf_score, r.matched_ingredients) for r in recipes],
        )
        return recipes


def build_retriever(settings: Settings) -> HaystackRecipeRetriever:
    """Compose provider, repository and pipeline into a retriever."""
    provider = resolve_provider(settings)
    repository = RecipeRepository(settings, provider)
    return HaystackRecipeRetriever(
        build_retrieval_pipeline(settings, provider, repository), top_k=settings.retriever_top_k
    )


class HaystackRecipeGenerator:
    """Runs the generation pipeline and returns Markdown."""

    def __init__(self, pipeline: AsyncPipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, ingredients: str, recipes: list[RetrievedRecipe]) -> str:
        result = await _run_async(
            self._pipeline,
            {"prompt_builder": {"ingredients": ingredients, "recipes": recipes}},
            label="generation",
        )
        replies = cast(list[ChatMessage], result["llm"]["replies"])
        if not replies or replies[0].text is None:
            raise RecommendationGenerationError("The LLM returned no recipe text")
        if replies[0].meta.get("finish_reason") == "length":
            # Half a recipe is worse than an error: the missing part is the steps.
            raise RecommendationGenerationError("The recipe answer was cut off at the token limit")
        return replies[0].text


_INTENT_KWARGS = {"max_tokens": 200, "temperature": 0}


def _parse_verdict(reply: str) -> IntentVerdict | None:
    """The gate's JSON reply, tolerating a stray code fence; `None` if it is not the schema."""
    body = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return IntentVerdict.model_validate_json(body)
    except ValidationError:
        return None


class HaystackIntentClassifier:
    """LLM classification that also names the ingredients; fails open on errors."""

    def __init__(self, pipeline: AsyncPipeline) -> None:
        self._pipeline = pipeline

    async def classify(self, text: str) -> IntentVerdict:
        try:
            result = await _run_async(
                self._pipeline,
                {"prompt_builder": {"text": text}, "llm": {"generation_kwargs": _INTENT_KWARGS}},
                label="intent",
            )
        except RecipeError as exc:
            # An outage in a guardrail must not take the endpoint down.
            logger.warning("Intent gate unavailable, allowing request: %s", exc)
            return IntentVerdict(is_ingredients=True)
        replies = cast(list[ChatMessage], result["llm"]["replies"])
        reply = (replies[0].text or "") if replies else ""
        verdict = _parse_verdict(reply)
        if verdict is None:
            logger.warning("Intent gate returned %r, allowing request", reply)
            return IntentVerdict(is_ingredients=True)
        return verdict


def build_intent_classifier(settings: Settings) -> HaystackIntentClassifier | None:
    """The gate on the provider's small model, or `None` when `INTENT_GATE_ENABLED=false`."""
    if not settings.intent_gate_enabled:
        return None
    return HaystackIntentClassifier(build_intent_pipeline(resolve_provider(settings)))
