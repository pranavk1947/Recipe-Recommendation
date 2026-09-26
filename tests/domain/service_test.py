"""Tests for the request flow. Fakes at every Protocol seam: no network, no database."""

from typing import cast

import pytest

from whats_for_dinner.core.exceptions import (
    OffTopicInputError,
    RecipeIndexEmptyError,
)
from whats_for_dinner.core.schemas import IntentVerdict, ParsedQuery, RetrievedRecipe
from whats_for_dinner.domain.service import RecommendationService

RECIPE = RetrievedRecipe(
    title="Tomato Soup",
    source_file="01.txt",
    vector_score=0.6,
    keyword_score=1.0,
    rrf_score=1.0,
    content="tomatoes, basil",
)


class FakeIntent:
    def __init__(self, allowed: bool = True, ingredients: list[str] | None = None) -> None:
        self.allowed = allowed
        self.ingredients = ingredients or []
        self.calls: list[str] = []

    async def classify(self, text: str) -> IntentVerdict:
        self.calls.append(text)
        return IntentVerdict(is_ingredients=self.allowed, ingredients=self.ingredients)


class FakeRetriever:
    def __init__(self, recipes: list[RetrievedRecipe]) -> None:
        self.recipes = recipes
        self.queries: list[ParsedQuery] = []

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedRecipe]:
        self.queries.append(query)
        return self.recipes


class FakeGenerator:
    def __init__(self, markdown: str = "# Tomato Soup") -> None:
        self.markdown = markdown
        self.seen: list[tuple[str, list[RetrievedRecipe]]] = []

    async def generate(self, ingredients: str, recipes: list[RetrievedRecipe]) -> str:
        self.seen.append((ingredients, recipes))
        return self.markdown


class FakeExtractor:
    def __init__(self, detected: list[str] | None = None) -> None:
        self.detected = detected if detected is not None else ["tomatoes", "basil"]

    async def extract(self, image: bytes) -> list[str]:
        return self.detected


_DEFAULT = object()


def build_service(
    *,
    intent: FakeIntent | object | None = _DEFAULT,
    retriever: FakeRetriever | None = None,
    generator: FakeGenerator | None = None,
    extractor: FakeExtractor | None = None,
) -> RecommendationService:
    return RecommendationService(
        cast(FakeIntent | None, FakeIntent() if intent is _DEFAULT else intent),
        retriever or FakeRetriever([RECIPE]),
        generator or FakeGenerator(),
        llm_model="gpt-4o",
        extractor=extractor,
    )


@pytest.mark.anyio
async def test_happy_path_returns_matches_without_recipe_bodies() -> None:
    retriever, generator = FakeRetriever([RECIPE]), FakeGenerator()
    intent = FakeIntent(ingredients=["tomatoes", "basil"])
    service = build_service(intent=intent, retriever=retriever, generator=generator)

    response = await service.recommend("2 large tomatoes and a bunch of basil")

    # The gate's plain names drive keyword search; the embedder still sees the raw text.
    assert retriever.queries[0].terms == ["tomatoes", "basil"]
    assert retriever.queries[0].text == "2 large tomatoes and a bunch of basil"

    assert response.recipe_markdown == "# Tomato Soup"
    assert response.llm_model == "gpt-4o"
    assert response.detected_ingredients is None
    assert response.image_status is None
    assert response.matched_recipes == [RECIPE.as_match()]
    assert "content" not in response.matched_recipes[0].model_dump()
    assert generator.seen[0][1] == [RECIPE]


@pytest.mark.anyio
async def test_off_topic_input_never_reaches_retrieval_or_generation() -> None:
    # The gate runs first, because retrieval needs the ingredient names it returns.
    retriever, generator = FakeRetriever([RECIPE]), FakeGenerator()
    service = build_service(
        intent=FakeIntent(allowed=False), retriever=retriever, generator=generator
    )

    with pytest.raises(OffTopicInputError):
        await service.recommend("what is the capital of france?")

    assert retriever.queries == []
    assert generator.seen == []


@pytest.mark.anyio
async def test_empty_index_is_reported_before_paying_for_the_llm() -> None:
    generator = FakeGenerator()
    service = build_service(retriever=FakeRetriever([]), generator=generator)

    with pytest.raises(RecipeIndexEmptyError, match="ingest"):
        await service.recommend("tomatoes")

    assert generator.seen == []


@pytest.mark.anyio
async def test_photo_ingredients_are_kept_however_unrelated_the_text_is() -> None:
    # The gate once saw text and photo together and dropped an unrelated photo as noise.
    intent, retriever = FakeIntent(ingredients=["chicken", "rice"]), FakeRetriever([RECIPE])
    service = build_service(intent=intent, retriever=retriever, extractor=FakeExtractor())

    await service.recommend("chicken breasts and rice", image=b"fake-image-bytes")

    assert intent.calls == ["chicken breasts and rice"]  # the gate reads typed text only
    assert retriever.queries[0].terms == ["chicken", "rice", "tomatoes", "basil"]


@pytest.mark.anyio
async def test_gibberish_text_with_a_food_photo_is_answered_from_the_photo() -> None:
    retriever = FakeRetriever([RECIPE])
    service = build_service(
        intent=FakeIntent(allowed=False), retriever=retriever, extractor=FakeExtractor()
    )

    await service.recommend("asdkj qwpoe", image=b"fake-image-bytes")

    assert retriever.queries[0].terms == ["tomatoes", "basil"]
