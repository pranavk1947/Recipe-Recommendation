"""The seams of the application: every boundary the implementations plug into.

Attributes rather than properties, so an implementation can just assign them.
"""

from typing import Protocol, runtime_checkable

from haystack import Document
from haystack.core.component import Component

from whats_for_dinner.core.schemas import IntentVerdict, ParsedQuery, RetrievedRecipe


@runtime_checkable
class IngredientExtractor(Protocol):
    """Reads ingredients off a photo. An empty list means no food was visible."""

    async def extract(self, image: bytes) -> list[str]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """One vendor: the Haystack components the pipelines need, plus the facts storage needs.

    To add a vendor, implement this and register it in `providers.PROVIDERS`.
    Return `None` from `ingredient_extractor()` if the vendor has no vision support.
    """

    name: str
    chat_model: str
    intent_model: str
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int

    def text_embedder(self) -> Component:
        """In `text` → out `embedding`."""
        ...

    def document_embedder(self) -> Component:
        """In `documents` → out `documents`."""
        ...

    def chat_generator(self, *, model: str | None = None) -> Component:
        """In `messages` → out `replies`. `model` defaults to `chat_model`."""
        ...

    def ingredient_extractor(self) -> IngredientExtractor | None: ...


@runtime_checkable
class IntentClassifier(Protocol):
    """The gate in front of retrieval. `is_ingredients=False` means off-topic → HTTP 400.

    Also names the ingredients, which become the keyword-search terms.
    """

    async def classify(self, text: str) -> IntentVerdict: ...


@runtime_checkable
class RecipeRetriever(Protocol):
    """Hybrid search over the cookbook."""

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedRecipe]: ...


@runtime_checkable
class RecipeGenerator(Protocol):
    """Turns candidates into the Markdown answer."""

    async def generate(self, ingredients: str, recipes: list[RetrievedRecipe]) -> str: ...


@runtime_checkable
class RecipeIndexer(Protocol):
    """Writes embedded recipes to the store; returns how many were written."""

    table_name: str

    def index(self, documents: list[Document]) -> int: ...


@runtime_checkable
class CoverageSearch(Protocol):
    """What `IngredientKeywordRetriever` needs: recipes ranked by how many terms they mention.

    Each returned document must carry `meta["matched_terms"]`, the terms it matched: the
    retriever reads it for `matched_ingredients`.
    """

    async def coverage_search(self, terms: list[str], top_k: int) -> list[Document]: ...
