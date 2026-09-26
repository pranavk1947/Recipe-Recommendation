"""Composition root: builds everything once at startup and wires it into FastAPI."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from whats_for_dinner.api.routes import router
from whats_for_dinner.core.config import Settings, configure_logging
from whats_for_dinner.core.exceptions import RecipeError
from whats_for_dinner.core.protocols import LLMProvider
from whats_for_dinner.core.schemas import ErrorResponse
from whats_for_dinner.domain.service import RecommendationService
from whats_for_dinner.providers import resolve_provider
from whats_for_dinner.rag.adapters import (
    HaystackIntentClassifier,
    HaystackRecipeGenerator,
    HaystackRecipeRetriever,
)
from whats_for_dinner.rag.pipelines import (
    build_generation_pipeline,
    build_intent_pipeline,
    build_retrieval_pipeline,
)
from whats_for_dinner.storage.repository import RecipeRepository

logger = logging.getLogger(__name__)


class AppComponents(BaseModel):
    """Everything the app needs, built once."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    settings: Settings
    provider: LLMProvider
    repository: RecipeRepository
    retriever: HaystackRecipeRetriever
    service: RecommendationService


def build_components(settings: Settings) -> AppComponents:
    """Production wiring. Reused by the integration test, so it cannot drift from reality."""
    provider = resolve_provider(settings)
    repository = RecipeRepository(settings, provider)
    retriever = HaystackRecipeRetriever(
        build_retrieval_pipeline(settings, provider, repository), top_k=settings.retriever_top_k
    )
    generator = HaystackRecipeGenerator(build_generation_pipeline(provider))
    intent = (
        HaystackIntentClassifier(build_intent_pipeline(provider))
        if settings.intent_gate_enabled
        else None
    )
    service = RecommendationService(
        intent,
        retriever,
        generator,
        llm_model=provider.chat_model,
        extractor=provider.ingredient_extractor(),
    )
    return AppComponents(
        settings=settings,
        provider=provider,
        repository=repository,
        retriever=retriever,
        service=service,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Build components, fail fast if Postgres is down, warn if nothing is indexed."""
    settings = Settings()
    configure_logging(settings.log_level)
    components = build_components(settings)  # ProviderConfigurationError surfaces here
    indexed = await components.repository.count()  # fails fast if Postgres is unreachable
    await asyncio.to_thread(components.retriever.warm_up)
    logger.info(
        "Provider %s (%s); table %s has %d recipes; intent gate %s",
        components.provider.name,
        components.provider.chat_model,
        components.repository.table_name,
        indexed,
        "on" if components.service.gate_enabled else "OFF",
    )
    if indexed == 0:
        logger.warning("Recipe index is empty - run `uv run ingest`")
    app.state.components = components
    yield
    await components.repository.aclose()


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="What's for Dinner",
        description="Recommends one cookbook recipe for the ingredients you have.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(RecipeError)
    async def handle_recipe_error(request: Request, exc: RecipeError) -> JSONResponse:
        """Turn any domain error into its declared status code."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
        )

    return app


app = create_app()
