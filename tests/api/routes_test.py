"""Tests for the HTTP surface. The service is faked; no network, no database."""

import httpx
import pytest
from fastapi import FastAPI

from whats_for_dinner.api.dependencies import get_service, get_settings
from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import (
    ImageInputUnavailableError,
    NoIngredientsProvidedError,
    RecipeError,
)
from whats_for_dinner.core.schemas import ErrorResponse, RecipeMatch, RecommendResponse
from whats_for_dinner.main import create_app

RESPONSE = RecommendResponse(
    recipe_markdown="# Tomato Soup",
    matched_recipes=[
        RecipeMatch(
            title="Tomato Soup",
            source_file="01.txt",
            vector_score=0.6,
            keyword_score=1.0,
            rrf_score=1.0,
        )
    ],
    llm_model="gpt-4o",
)


class FakeService:
    def __init__(self, error: RecipeError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, bytes | None]] = []

    async def recommend(
        self, ingredients_text: str, image: bytes | None = None
    ) -> RecommendResponse:
        self.calls.append((ingredients_text, image))
        if self.error is not None:
            raise self.error
        return RESPONSE


def build_client(service: FakeService, settings: Settings) -> tuple[FastAPI, httpx.AsyncClient]:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: settings
    transport = httpx.ASGITransport(app=app)
    return app, httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.anyio
async def test_form_post_returns_the_recommendation(settings: Settings) -> None:
    service = FakeService()
    _, client = build_client(service, settings)

    async with client:
        response = await client.post("/recommend_recipe", data={"ingredients": "tomatoes"})

    assert response.status_code == 200
    body = response.json()
    assert body["recipe_markdown"] == "# Tomato Soup"
    assert "content" not in body["matched_recipes"][0]  # recipe bodies stay server-side
    assert service.calls == [("tomatoes", None)]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NoIngredientsProvidedError("nothing"), 400),
        (ImageInputUnavailableError("no vision"), 501),
    ],
)
async def test_domain_errors_keep_their_status_and_name(
    settings: Settings, error: RecipeError, status: int
) -> None:
    _, client = build_client(FakeService(error), settings)

    async with client:
        response = await client.post("/recommend_recipe", data={"ingredients": "x"})

    assert response.status_code == status
    body = ErrorResponse.model_validate(response.json())
    assert body.error == type(error).__name__


@pytest.mark.anyio
async def test_an_image_reaches_the_service(settings: Settings) -> None:
    service = FakeService()
    _, client = build_client(service, settings)

    async with client:
        response = await client.post(
            "/recommend_recipe",
            data={"ingredients": ""},
            files={"image": ("food.webp", b"image-bytes", "image/webp")},
        )

    assert response.status_code == 200
    assert service.calls == [("", b"image-bytes")]
