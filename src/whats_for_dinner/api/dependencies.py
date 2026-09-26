"""FastAPI dependencies: the components built at startup, and HTTP request parsing."""

from typing import Annotated

from fastapi import Depends, File, Request, UploadFile

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import InvalidImageError, UnsupportedMediaTypeError
from whats_for_dinner.core.protocols import LLMProvider
from whats_for_dinner.domain.service import RecommendationService
from whats_for_dinner.storage.repository import RecipeRepository

_FORM_MEDIA_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")


def get_service(request: Request) -> RecommendationService:
    """The service built once during startup."""
    return request.app.state.components.service


def get_settings(request: Request) -> Settings:
    return request.app.state.components.settings


def get_repository(request: Request) -> RecipeRepository:
    return request.app.state.components.repository


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.components.provider


ServiceDep = Annotated[RecommendationService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RepositoryDep = Annotated[RecipeRepository, Depends(get_repository)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]


def ensure_form_request(request: Request) -> None:
    """Reject JSON bodies explicitly: a 415 beats the empty-form 400 they would otherwise get."""
    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.startswith(_FORM_MEDIA_TYPES):
        raise UnsupportedMediaTypeError(
            "Send form data (multipart/form-data or application/x-www-form-urlencoded), not JSON"
        )


async def read_image(
    settings: SettingsDep, image: Annotated[UploadFile | str | None, File()] = None
) -> bytes | None:
    """The uploaded photo's bytes, size-guarded. No content-type check: Pillow decides.

    Swagger UI posts an empty string for an unfilled file field, so an empty value counts as
    "no image" rather than a 422 from the form parser. The union annotation makes pydantic
    hand back a starlette `UploadFile`, which is not an instance of FastAPI's subclass, so
    narrow on `str` rather than testing for `UploadFile`.
    """
    if image is None or isinstance(image, str):
        return None
    max_bytes = settings.max_image_bytes
    if image.size is not None and image.size > max_bytes:
        raise InvalidImageError(f"Image larger than {max_bytes} bytes")
    data = await image.read()
    if not data or len(data) > max_bytes:
        raise InvalidImageError("Image is empty or too large")
    return data


ImageDep = Annotated[bytes | None, Depends(read_image)]
