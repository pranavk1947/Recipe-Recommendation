"""Pytest configuration and shared fixtures."""

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from whats_for_dinner.core.config import Settings


class IsolatedSettings(Settings):
    """Settings that ignore the developer's real `.env` so tests are reproducible."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", env_ignore_empty=True)


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio to asyncio; without this anyio also runs every test under trio."""
    return "asyncio"


@pytest.fixture
def isolated_settings_cls(monkeypatch: pytest.MonkeyPatch) -> type[IsolatedSettings]:
    """The isolated Settings class with every one of its env vars cleared."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return IsolatedSettings


@pytest.fixture
def settings(isolated_settings_cls: type[IsolatedSettings]) -> Settings:
    """Default settings with a dummy API key, isolated from the real environment."""
    return isolated_settings_cls(openai_api_key=SecretStr("test-key"))
