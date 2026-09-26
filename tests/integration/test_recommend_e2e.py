"""End-to-end test against a real Postgres and a real LLM provider.

Opt-in: needs Docker running, an ingested table and a working API key.

    docker compose up -d db
    RUN_INTEGRATION=1 uv run pytest -m integration
"""

import asyncio
import gc
import os
from collections.abc import AsyncIterator

import pytest

from whats_for_dinner.cli.ingest import run_ingestion
from whats_for_dinner.core.config import Settings
from whats_for_dinner.main import build_components
from whats_for_dinner.providers import resolve_provider
from whats_for_dinner.rag.adapters import build_indexer
from whats_for_dinner.storage.repository import RecipeRepository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"
    ),
]


@pytest.fixture(autouse=True)
async def close_clients_on_this_loop() -> AsyncIterator[None]:
    """OpenAI's async client closes itself from `__del__` on whatever loop is running then.

    Left to chance, that is the next test's loop, which fails with "Event loop is closed".
    Collecting here, while this test's loop is still alive, closes them where they belong.
    """
    yield
    gc.collect()
    await asyncio.sleep(0.1)


@pytest.mark.anyio
async def test_a_real_request_returns_a_grounded_recipe() -> None:
    settings = Settings()
    # Production wiring, so this cannot drift from what the app actually builds.
    components = build_components(settings)
    run_ingestion(settings.recipes_dir, build_indexer(settings))

    response = await components.service.recommend("chicken breasts, spinach, feta")

    assert response.recipe_markdown.startswith("#")
    assert "Based on:" in response.recipe_markdown
    assert response.matched_recipes
    top = response.matched_recipes[0]
    assert "spinach" in top.matched_ingredients
    # The answer must name a recipe we actually retrieved, not an invented one.
    titles = [match.title.lower() for match in response.matched_recipes]
    assert any(title in response.recipe_markdown.lower() for title in titles)


@pytest.mark.anyio
async def test_off_topic_input_is_refused_before_retrieval() -> None:
    from whats_for_dinner.core.exceptions import OffTopicInputError

    components = build_components(Settings())

    with pytest.raises(OffTopicInputError):
        await components.service.recommend("What is the capital of France?")


@pytest.mark.anyio
async def test_coverage_is_counted_in_one_query_without_embeddings() -> None:
    settings = Settings()
    repository = RecipeRepository(settings, resolve_provider(settings))

    hits = await repository.coverage_search(["eggs", "spinach", "feta"], top_k=3)

    assert len(hits) == 3
    # Two recipes contain two of the three; nothing contains all three.
    assert {hit.meta["title"] for hit in hits[:2]} == {
        "Spinach and Feta Stuffed Chicken",
        "Vegetable Frittata",
    }
    assert [hit.score for hit in hits][:2] == [pytest.approx(2 / 3)] * 2
    assert all(hit.embedding is None for hit in hits)
    # The same Postgres match names the terms, in the order given, so they agree with the count.
    assert {tuple(hit.meta["matched_terms"]) for hit in hits[:2]} == {
        ("spinach", "feta"),
        ("eggs", "spinach"),
    }
