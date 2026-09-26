"""Turn the intent gate's ingredient names into search terms."""

from whats_for_dinner.core.schemas import ParsedQuery

_MAX_TERMS = 20


def parse_ingredients_query(text: str, ingredients: list[str]) -> ParsedQuery:
    """Build the query. The embedder always sees the raw text; `terms` drive keyword search.

    `ingredients` are the names the intent gate extracted. With none (gate off or failed),
    keyword search sits out and vector search alone ranks the recipes.
    """
    terms: list[str] = []
    for ingredient in ingredients:
        term = " ".join(ingredient.lower().split())
        if term and term not in terms:
            terms.append(term)
    return ParsedQuery(text=text.strip(), terms=terms[:_MAX_TERMS])
