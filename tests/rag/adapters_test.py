"""Tests for the adapters: error translation, parsing and the fail-open intent gate."""

import httpx
import openai
import pytest
from haystack import AsyncPipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.errors import DocumentStoreError

from whats_for_dinner.core.exceptions import (
    ProviderConfigurationError,
    RecipeStoreUnavailableError,
    RecommendationGenerationError,
)
from whats_for_dinner.core.schemas import IntentVerdict
from whats_for_dinner.rag.adapters import (
    HaystackIntentClassifier,
    HaystackRecipeGenerator,
    translate_pipeline_error,
)

API_ERROR = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        # Haystack may wrap component failures; the mapping must survive that.
        (RuntimeError("pipeline blew up"), RecommendationGenerationError),
        (DocumentStoreError("no connection"), RecipeStoreUnavailableError),
        # A wrong key is a configuration fault (500), not an upstream outage (502).
        (
            openai.AuthenticationError(
                "bad key", response=httpx.Response(401, request=API_ERROR.request), body=None
            ),
            ProviderConfigurationError,
        ),
    ],
)
def test_known_failures_become_domain_errors(raised: Exception, expected: type[Exception]) -> None:
    if isinstance(raised, RuntimeError):
        raised.__cause__ = API_ERROR
    translated = translate_pipeline_error(raised)
    assert isinstance(translated, expected)


class FakeLLMPipeline(AsyncPipeline):
    """Stands in for prompt_builder → llm, recording what it was called with."""

    def __init__(self, replies: list[ChatMessage] | None = None, error: Exception | None = None):
        self.seen: dict[str, dict[str, object]] = {}
        self._replies = replies if replies is not None else []
        self._error = error

    async def run_async(
        self, data: object, include_outputs_from: object = None, concurrency_limit: int = 4
    ) -> dict[str, dict[str, object]]:
        if isinstance(data, dict):
            self.seen = data
        if self._error is not None:
            raise self._error
        return {"llm": {"replies": self._replies}}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '{"is_ingredients": true, "ingredients": ["eggs", "rice"]}',
            IntentVerdict(is_ingredients=True, ingredients=["eggs", "rice"]),
        ),
        # Unparseable → fail open, with no names: keyword search sits out.
        ("INGREDIENTS", IntentVerdict(is_ingredients=True)),
    ],
)
async def test_intent_gate_verdicts(reply: str, expected: IntentVerdict) -> None:
    pipeline = FakeLLMPipeline([ChatMessage.from_assistant(reply)])
    classifier = HaystackIntentClassifier(pipeline)

    assert await classifier.classify("eggs and rice") == expected
    # Deterministic, and room for up to 20 ingredient names.
    assert pipeline.seen["llm"] == {"generation_kwargs": {"max_tokens": 200, "temperature": 0}}


@pytest.mark.anyio
async def test_intent_gate_fails_open_when_the_llm_is_down() -> None:
    # A guardrail outage must not take the endpoint down.
    classifier = HaystackIntentClassifier(FakeLLMPipeline(error=API_ERROR))
    assert await classifier.classify("eggs and rice") == IntentVerdict(is_ingredients=True)


@pytest.mark.anyio
async def test_a_truncated_recipe_is_an_error_not_a_half_answer() -> None:
    cut_off = ChatMessage.from_assistant("# Soup\n1. Heat the", meta={"finish_reason": "length"})
    generator = HaystackRecipeGenerator(FakeLLMPipeline([cut_off]))

    with pytest.raises(RecommendationGenerationError, match="cut off"):
        await generator.generate("eggs", [])
