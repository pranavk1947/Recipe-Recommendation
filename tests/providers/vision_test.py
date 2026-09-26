"""Tests for photo ingredient extraction. The OpenAI client is mocked."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from whats_for_dinner.providers.vision import OpenAIVisionIngredientExtractor

JPEG_MAGIC = b"\xff\xd8\xff"


def make_image(fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA" if fmt == "PNG" else "RGB", (8, 8), "red").save(buffer, format=fmt)
    return buffer.getvalue()


def build_extractor(
    content: str | None = '{"ingredients": ["eggs", "spinach"]}',
) -> tuple[OpenAIVisionIngredientExtractor, AsyncMock]:
    client = AsyncMock()
    completion = AsyncMock()
    completion.choices = [AsyncMock(message=AsyncMock(content=content))]
    # Real usage numbers: the extractor logs tokens from them.
    completion.usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8)
    client.chat.completions.create = AsyncMock(return_value=completion)
    return OpenAIVisionIngredientExtractor(client, "gpt-4o"), client


@pytest.mark.anyio
async def test_it_returns_the_detected_ingredients() -> None:
    extractor, client = build_extractor('{"ingredients": ["eggs", " spinach "]}')

    assert await extractor.extract(make_image("PNG")) == ["eggs", "spinach"]

    sent = client.chat.completions.create.await_args.kwargs
    image_part = sent["messages"][0]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_part["image_url"]["detail"] == "low"


@pytest.mark.anyio
async def test_a_photo_without_food_yields_nothing() -> None:
    extractor, _ = build_extractor('{"ingredients": []}')
    assert await extractor.extract(make_image("PNG")) == []
