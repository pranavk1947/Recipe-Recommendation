"""Haystack pipeline builders. Each returns wiring only - no error handling, no I/O."""

from haystack import AsyncPipeline, Pipeline
from haystack.components.builders import ChatPromptBuilder

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.protocols import LLMProvider
from whats_for_dinner.rag.components import IngredientKeywordRetriever
from whats_for_dinner.rag.prompts import INTENT_PROMPT, RECIPE_PROMPT
from whats_for_dinner.storage.repository import RecipeRepository


def build_indexing_pipeline(provider: LLMProvider, repository: RecipeRepository) -> Pipeline:
    """Embed recipes and write them to the store. Sync: ingestion is a one-shot command."""
    pipeline = Pipeline()
    pipeline.add_component("embedder", provider.document_embedder())
    pipeline.add_component("writer", repository.document_writer())
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def build_retrieval_pipeline(
    settings: Settings, provider: LLMProvider, repository: RecipeRepository
) -> AsyncPipeline:
    """Hybrid retrieval: vector search and ingredient-coverage keyword search, side by side.

    Both candidate lists come out of the pipeline; the adapter fuses them (`fuse_by_rank`).
    """
    pipeline = AsyncPipeline()
    pipeline.add_component("text_embedder", provider.text_embedder())
    pipeline.add_component(
        "embedding_retriever", repository.embedding_retriever(settings.retriever_candidates)
    )
    pipeline.add_component(
        "keyword_retriever",
        IngredientKeywordRetriever(repository, top_k=settings.retriever_candidates),
    )
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    return pipeline


def build_generation_pipeline(provider: LLMProvider) -> AsyncPipeline:
    """Render the recipe prompt and send it to the provider's chat model."""
    pipeline = AsyncPipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=RECIPE_PROMPT, required_variables=["ingredients", "recipes"]),
    )
    pipeline.add_component("llm", provider.chat_generator())
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def build_intent_pipeline(provider: LLMProvider) -> AsyncPipeline:
    """Classify the request and name its ingredients, on the provider's cheap model.

    This runs before retrieval on every request, so a small model keeps it fast.
    Can also explore Jev for intent classification
    """
    pipeline = AsyncPipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=INTENT_PROMPT, required_variables=["text"]),
    )
    pipeline.add_component("llm", provider.chat_generator(model=provider.intent_model))
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline
