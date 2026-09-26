"""Read ingredients off a photo with gpt-4o vision, via the OpenAI SDK.

Haystack 2.12 has no image content type for chat messages, so this talks to the SDK
directly and stays behind the `IngredientExtractor` Protocol.
"""

import asyncio
import base64
import io
import logging
from time import perf_counter

from openai import AsyncOpenAI, AuthenticationError, OpenAIError
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from whats_for_dinner.core.exceptions import (
    IngredientExtractionError,
    InvalidImageError,
    ProviderConfigurationError,
)

logger = logging.getLogger(__name__)

_PROMPT = (
    "List the food ingredients visible in this photo, each by its plain name. Only list items "
    'you can actually see. Reply with JSON only: {"ingredients": ["...", "..."]}, with an empty '
    "list if there is no food."
)


class _VisionReply(BaseModel):
    """The same JSON contract as the intent gate's, so neither extraction parses free text."""

    ingredients: list[str]


def to_jpeg_base64(image: bytes) -> str:
    """Re-encode any Pillow-readable image as base64 JPEG.

    Args:
        image: Raw uploaded bytes, in any format Pillow can decode.

    Returns:
        Base64-encoded JPEG data, without a data-URL prefix.

    Raises:
        InvalidImageError: The bytes could not be decoded as an image.
    """
    try:
        with Image.open(io.BytesIO(image)) as opened:
            # JPEG has no alpha channel, and RGB keeps the request small.
            converted = opened.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=85)
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidImageError(f"Could not read the image: {exc}") from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAIVisionIngredientExtractor:
    """Names the ingredients in a photo. Returns an empty list when there is no food in it."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def extract(self, image: bytes) -> list[str]:
        """Describe the ingredients visible in a photo.

        Args:
            image: Raw image bytes from the upload.

        Returns:
            The ingredient names, or an empty list if no food is visible.

        Raises:
            InvalidImageError: The bytes are not a decodable image.
            IngredientExtractionError: The vision request failed or its reply was not the JSON.
        """
        encoded = await asyncio.to_thread(to_jpeg_base64, image)
        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {
                    "type": "image_url",
                    # "low" detail is enough to name ingredients and costs far fewer tokens.
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "low",
                    },
                },
            ],
        }
        started = perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[message],
                max_tokens=200,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise ProviderConfigurationError(
                "OpenAI rejected the API key - check OPENAI_API_KEY"
            ) from exc
        except OpenAIError as exc:
            raise IngredientExtractionError(f"Vision request failed: {exc}") from exc
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        logger.info(
            "step=vision.llm model=%s duration=%.2fs tokens=%d+%d",
            self._model,
            perf_counter() - started,
            prompt_tokens,
            completion_tokens,
        )

        content = completion.choices[0].message.content if completion.choices else None
        try:
            reply = _VisionReply.model_validate_json(content or "")
        except ValidationError as exc:
            raise IngredientExtractionError(f"Vision reply was not the JSON: {content!r}") from exc
        ingredients = [name.strip() for name in reply.ingredients if name.strip()]
        if not ingredients:
            logger.info("No food visible in the uploaded photo")
        return ingredients
