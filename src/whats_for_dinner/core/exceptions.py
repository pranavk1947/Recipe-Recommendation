"""Domain exceptions. The HTTP status lives on the class, so the API needs one handler."""


class RecipeError(Exception):
    """Base for every error this service turns into an HTTP response."""

    status_code: int = 500


class NoIngredientsProvidedError(RecipeError):
    """Neither usable ingredient text nor an image was supplied."""

    status_code = 400


class OffTopicInputError(RecipeError):
    """The intent gate judged the input not to be a list of ingredients."""

    status_code = 400


class InvalidImageError(RecipeError):
    """The uploaded file was too large or could not be decoded."""

    status_code = 400


class UnsupportedMediaTypeError(RecipeError):
    """The request body was not a form (a JSON body is the usual cause)."""

    status_code = 415


class ImageInputUnavailableError(RecipeError):
    """The configured provider has no vision support."""

    status_code = 501


class RecommendationGenerationError(RecipeError):
    """The LLM call for the recipe answer failed."""

    status_code = 502


class IngredientExtractionError(RecipeError):
    """The vision call failed."""

    status_code = 502


class RecipeIndexEmptyError(RecipeError):
    """The recipe table exists but holds no documents; ingestion has not run."""

    status_code = 503


class RecipeStoreUnavailableError(RecipeError):
    """Postgres could not be reached."""

    status_code = 503


class ProviderConfigurationError(RecipeError):
    """The provider is missing configuration; raised at startup, not per request."""

    status_code = 500


class RecipeParseError(RecipeError):
    """A recipe file could not be read or did not have the expected sections."""

    status_code = 500
