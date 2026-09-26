"""The provider registry: the one place a new LLM vendor is wired in."""

from collections.abc import Callable

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import ProviderConfigurationError
from whats_for_dinner.core.protocols import LLMProvider
from whats_for_dinner.providers.openai_provider import OpenAIProvider

# Adding a provider = one line here (plus its module and its env vars in Settings).
PROVIDERS: dict[str, Callable[[Settings], LLMProvider]] = {"openai": OpenAIProvider}


def resolve_provider(settings: Settings) -> LLMProvider:
    """Build the provider named by `LLM_PROVIDER`."""
    factory = PROVIDERS.get(settings.llm_provider)
    if factory is None:
        raise ProviderConfigurationError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}; available: {sorted(PROVIDERS)}"
        )
    return factory(settings)


__all__ = ["PROVIDERS", "OpenAIProvider", "resolve_provider"]
