"""Pydantic models at the application's boundaries."""

from typing import Literal

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    """The user's ingredient text, plus the individual terms pulled out of it."""

    text: str
    terms: list[str] = Field(default_factory=list)


class IntentVerdict(BaseModel):
    """The gate's reply: is this an ingredient list, and which ingredients does it name."""

    is_ingredients: bool
    ingredients: list[str] = Field(default_factory=list)


class RecipeMatch(BaseModel):
    """A retrieved recipe as the API reports it."""

    title: str
    source_file: str
    vector_score: float | None = Field(
        description="Cosine similarity to your text; null if not in the vector top candidates."
    )
    keyword_score: float | None = Field(
        description="Fraction of your ingredients the recipe contains; null if not in the "
        "keyword top candidates."
    )
    rrf_score: float = Field(
        description="Reciprocal rank fusion of the two rankings - the order you see. "
        "1.0 = ranked first by both, 0.5 = first by only one."
    )
    matched_ingredients: list[str] = Field(default_factory=list)


class RetrievedRecipe(RecipeMatch):
    """A `RecipeMatch` plus the recipe text the prompt needs."""

    content: str

    def as_match(self) -> RecipeMatch:
        """Drop the recipe body so the response stays small."""
        return RecipeMatch.model_validate(self.model_dump(exclude={"content"}))


# `None` means no photo was sent at all, which `detected_ingredients` alone cannot express:
# it is also `None` when a photo was sent and nothing edible was found in it.
ImageStatus = Literal["ingredients_detected", "no_food_detected"]


class RecommendResponse(BaseModel):
    """Body of a successful `POST /recommend_recipe`."""

    recipe_markdown: str
    matched_recipes: list[RecipeMatch]
    detected_ingredients: list[str] | None = None
    image_status: ImageStatus | None = None
    llm_model: str


class HealthResponse(BaseModel):
    """Body of `GET /health`."""

    status: Literal["ok"]
    provider: str
    llm_model: str
    embedding_model: str
    table: str
    indexed_recipes: int
    intent_gate: bool


class ErrorResponse(BaseModel):
    """Body of every error response."""

    error: str
    detail: str
