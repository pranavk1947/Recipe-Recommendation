"""Tests for recipe parsing."""

from pathlib import Path

import pytest

from whats_for_dinner.domain.recipes import load_recipes, parse_recipe, recipe_to_document

# A leading blank line (13 of the 20 files have one) and list markers (defensive).
SAMPLE = """
Quick Chicken Stir-Fry

Ingredients:

    1. 2 chicken breasts, diced
    - 1 tbsp vegetable oil

Instructions:

    1. Heat oil in a wok.
    2. Serve hot over rice.
"""


def test_parses_title_sections_and_strips_markers() -> None:
    recipe = parse_recipe(SAMPLE, "01.txt")
    assert recipe.title == "Quick Chicken Stir-Fry"
    assert recipe.ingredients == ["2 chicken breasts, diced", "1 tbsp vegetable oil"]
    assert recipe.instructions == ["Heat oil in a wok.", "Serve hot over rice."]

    document = recipe_to_document(recipe)
    assert document.id == "recipe:01"  # deterministic id makes re-ingestion idempotent
    assert document.meta == {"title": "Quick Chicken Stir-Fry", "source_file": "01.txt"}
    assert document.content is not None
    assert document.content.startswith("Quick Chicken Stir-Fry")


@pytest.mark.skipif(not Path("data/recipes").is_dir(), reason="data.zip not unzipped")
def test_the_real_cookbook_parses() -> None:
    recipes = load_recipes(Path("data/recipes"))
    assert len(recipes) == 20
    assert all(r.ingredients and r.instructions and r.title for r in recipes)
    assert [r.document_id for r in recipes][:2] == ["recipe:01", "recipe:02"]
