"""The request flow, expressed against Protocols so nothing here knows about Haystack."""

import asyncio
import logging

from whats_for_dinner.core.exceptions import (
    ImageInputUnavailableError,
    NoIngredientsProvidedError,
    OffTopicInputError,
    RecipeIndexEmptyError,
)
from whats_for_dinner.core.protocols import (
    IngredientExtractor,
    IntentClassifier,
    RecipeGenerator,
    RecipeRetriever,
)
from whats_for_dinner.core.schemas import (
    ImageStatus,
    IntentVerdict,
    RecommendResponse,
    RetrievedRecipe,
)
from whats_for_dinner.domain.query import parse_ingredients_query

logger = logging.getLogger(__name__)


def compose_query(ingredients_text: str, detected: list[str]) -> str:
    """Combine typed ingredients with anything spotted in a photo. User text comes first."""
    parts = [ingredients_text.strip()]
    if detected:
        parts.append(f"Ingredients visible in the photo: {', '.join(detected)}")
    return "\n".join(part for part in parts if part)


def is_grounded(markdown: str, recipes: list[RetrievedRecipe]) -> bool:
    """True if the answer names one of the candidates we actually retrieved."""
    return any(recipe.title.lower() in markdown.lower() for recipe in recipes)


class RecommendationService:
    """Guard, detect, gate, retrieve, generate, respond."""

    def __init__(
        self,
        intent: IntentClassifier | None,
        retriever: RecipeRetriever,
        generator: RecipeGenerator,
        *,
        llm_model: str,
        extractor: IngredientExtractor | None,
    ) -> None:
        self._intent = intent
        self._retriever = retriever
        self._generator = generator
        self._llm_model = llm_model
        self._extractor = extractor

    @property
    def gate_enabled(self) -> bool:
        """Whether off-topic input is refused rather than answered with the closest match."""
        return self._intent is not None

    async def recommend(
        self, ingredients_text: str, image: bytes | None = None
    ) -> RecommendResponse:
        """Recommend one cookbook recipe for the ingredients the user has.

        Args:
            ingredients_text: Free-text ingredients from the form.
            image: Optional photo of ingredients.

        Returns:
            The Markdown recommendation, the recipes it was based on, and what the photo
            contributed - `image_status` separates "no photo sent" from "photo had no food".

        Raises:
            NoIngredientsProvidedError: Nothing usable was supplied.
            OffTopicInputError: The text is not a list of ingredients and no photo showed food
                (gate enabled only).
            ImageInputUnavailableError: An image was sent but the provider has no vision.
            RecipeIndexEmptyError: Nothing is indexed yet.
        """
        if image is None and not ingredients_text.strip():
            raise NoIngredientsProvidedError("Provide ingredients as text, an image, or both")

        # Independent calls, so they run together. The photo's list goes straight into the
        # terms: passed through the gate beside unrelated text, the gate dropped it as noise.
        detected, verdict = await asyncio.gather(
            self._detect_ingredients(image), self._classify(ingredients_text)
        )
        image_status: ImageStatus | None = None
        if image is not None:
            image_status = "ingredients_detected" if detected else "no_food_detected"
        text = compose_query(ingredients_text, detected)
        if not text:
            raise NoIngredientsProvidedError(
                "No ingredients in the text and none visible in the photo"
            )
        # Refused in code, before retrieval and the recipe prompt - unless a photo showed food.
        if verdict is not None and not verdict.is_ingredients and not detected:
            raise OffTopicInputError(
                "This service only recommends recipes from a list of ingredients - "
                "describe what you have in your kitchen"
            )

        typed = verdict.ingredients if verdict is not None else []
        query = parse_ingredients_query(text, typed + detected)
        recipes = await self._retriever.retrieve(query)
        if not recipes:
            raise RecipeIndexEmptyError("No recipes are indexed - run `uv run ingest`")

        markdown = await self._generator.generate(query.text, recipes)
        if not is_grounded(markdown, recipes):
            logger.warning(
                "Possible grounding failure: no retrieved title in the answer (%s)",
                [recipe.title for recipe in recipes],
            )
        return RecommendResponse(
            recipe_markdown=markdown,
            matched_recipes=[recipe.as_match() for recipe in recipes],
            detected_ingredients=detected or None,
            image_status=image_status,
            llm_model=self._llm_model,
        )

    async def _classify(self, typed: str) -> IntentVerdict | None:
        """The gate's verdict on the typed text; `None` for empty text or a disabled gate."""
        if self._intent is None or not typed.strip():
            return None
        return await self._intent.classify(typed.strip())

    async def _detect_ingredients(self, image: bytes | None) -> list[str]:
        if image is None:
            return []
        if self._extractor is None:
            raise ImageInputUnavailableError("The active LLM provider does not support image input")
        return await self._extractor.extract(image)
