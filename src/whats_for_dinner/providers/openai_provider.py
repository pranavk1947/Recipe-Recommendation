"""OpenAI implementation of `LLMProvider`."""

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.component import Component
from haystack.utils import Secret
from openai import AsyncOpenAI

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import ProviderConfigurationError
from whats_for_dinner.core.protocols import IngredientExtractor
from whats_for_dinner.providers.vision import OpenAIVisionIngredientExtractor


class OpenAIProvider:
    """gpt-4o for chat and vision, text-embedding-3-small (1536) for embeddings."""

    name = "openai"
    embedding_backend = "openai"

    def __init__(self, settings: Settings) -> None:
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is not set (environment variable or .env)"
            )
        self._api_key = api_key
        self._secret = Secret.from_token(api_key)
        self.chat_model = settings.openai_chat_model
        self.intent_model = settings.openai_intent_model
        self.embedding_model = settings.openai_embedding_model
        self.embedding_dimension = settings.openai_embedding_dimension

    def text_embedder(self) -> Component:
        return OpenAITextEmbedder(api_key=self._secret, model=self.embedding_model)

    def document_embedder(self) -> Component:
        return OpenAIDocumentEmbedder(
            api_key=self._secret, model=self.embedding_model, progress_bar=False
        )

    def chat_generator(self, *, model: str | None = None) -> Component:
        return OpenAIChatGenerator(
            api_key=self._secret,
            model=model or self.chat_model,
            generation_kwargs={"temperature": 0.3, "max_tokens": 1000},
        )

    def ingredient_extractor(self) -> IngredientExtractor | None:
        # Haystack 2.12 has no image chat content, so vision goes through the SDK directly.
        return OpenAIVisionIngredientExtractor(AsyncOpenAI(api_key=self._api_key), self.chat_model)
