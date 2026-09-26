"""HTTP surface: endpoints only. Wiring and request parsing live in `dependencies.py`."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form

from whats_for_dinner.api.dependencies import (
    ImageDep,
    ProviderDep,
    RepositoryDep,
    ServiceDep,
    ensure_form_request,
)
from whats_for_dinner.core.schemas import ErrorResponse, HealthResponse, RecommendResponse

router = APIRouter()

# FastAPI keys this by `int | str`, and dict keys are invariant, so annotate explicitly.
_ERROR_RESPONSES: dict[int | str, dict[str, type[ErrorResponse]]] = {
    code: {"model": ErrorResponse} for code in (400, 415, 500, 501, 502, 503)
}


@router.post(
    "/recommend_recipe",
    response_model=RecommendResponse,
    responses=_ERROR_RESPONSES,
    dependencies=[Depends(ensure_form_request)],
)
async def recommend_recipe(
    service: ServiceDep,
    image: ImageDep,
    ingredients: Annotated[str, Form(max_length=4000)] = "",
) -> RecommendResponse:
    """Recommend one cookbook recipe for the ingredients supplied as text and/or a photo.

    Args:
        service: The recommendation service.
        image: The optional photo's bytes, already size-checked.
        ingredients: Free-text ingredients.

    Returns:
        The Markdown recommendation and the recipes behind it.
    """
    return await service.recommend(ingredients, image)


@router.get("/health", response_model=HealthResponse)
async def health(
    repository: RepositoryDep, provider: ProviderDep, service: ServiceDep
) -> HealthResponse:
    """Report configuration and how many recipes are indexed.

    Returns:
        The health payload. A probe is infrastructure, so it reads the repository directly.
    """
    return HealthResponse(
        status="ok",
        provider=provider.name,
        llm_model=provider.chat_model,
        embedding_model=provider.embedding_model,
        table=repository.table_name,
        indexed_recipes=await repository.count(),
        intent_gate=service.gate_enabled,
    )
