"""`uv run search "eggs, spinach"` — inspect retrieval without paying for the recipe answer.

Terms come from the intent gate, exactly as in the API, so this shows what a request would match.
"""

import asyncio
from collections.abc import Sequence

from whats_for_dinner.core.config import Settings, configure_logging
from whats_for_dinner.core.exceptions import RecipeError
from whats_for_dinner.core.schemas import IntentVerdict, ParsedQuery, RetrievedRecipe
from whats_for_dinner.domain.query import parse_ingredients_query
from whats_for_dinner.rag.adapters import build_intent_classifier, build_retriever


async def _search(settings: Settings, text: str) -> tuple[ParsedQuery, list[RetrievedRecipe]]:
    """Gate, then retrieve - the API's order, minus generation."""
    intent = build_intent_classifier(settings)
    verdict = await intent.classify(text) if intent else IntentVerdict(is_ingredients=True)
    if not verdict.is_ingredients:
        raise SystemExit("search refused: not an ingredient list (the API answers 400)")
    query = parse_ingredients_query(text, verdict.ingredients)
    return query, await build_retriever(settings).retrieve(query)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point. Everything after the command name is the ingredient text."""
    import sys

    words = list(argv) if argv is not None else sys.argv[1:]
    text = " ".join(words).strip()
    if not text:
        raise SystemExit('usage: uv run search "2 eggs, spinach, feta"')

    settings = Settings()
    configure_logging(settings.log_level)

    try:
        query, recipes = asyncio.run(_search(settings, text))
    except RecipeError as exc:
        raise SystemExit(f"search failed: {exc}") from exc

    print(f"provider={settings.llm_provider} terms={query.terms}")
    print("   rrf  vector keyword  recipe")
    for recipe in recipes:
        print(
            f"  {recipe.rrf_score:.3f} {_fmt(recipe.vector_score)} {_fmt(recipe.keyword_score)}  "
            f"{recipe.title} ({recipe.source_file}) matched={recipe.matched_ingredients}"
        )


def _fmt(score: float | None) -> str:
    """A score column; `-` when the recipe was not in that retriever's candidates."""
    return f"{score:6.3f}" if score is not None else "     -"


if __name__ == "__main__":
    main()
