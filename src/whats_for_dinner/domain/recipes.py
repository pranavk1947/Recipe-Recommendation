"""Read the cookbook: `data/recipes/*.txt` → `Recipe` → Haystack `Document`."""

import re
from pathlib import Path

from haystack import Document
from pydantic import BaseModel

from whats_for_dinner.core.exceptions import RecipeParseError

# A section header on a line of its own, e.g. "Ingredients:".
_HEADER = re.compile(r"^(?P<name>[A-Za-z][A-Za-z ]*):\s*$")
# Leading list markers: "1.", "2)", "-", "*", "•".
_MARKER = re.compile(r"^(\d+[.)]|[-*•])\s+")
_SECTIONS = ("ingredients", "instructions")


class Recipe(BaseModel):
    """One recipe file."""

    source_file: str
    title: str
    ingredients: list[str]
    instructions: list[str]

    @property
    def document_id(self) -> str:
        """Deterministic id, so re-ingesting overwrites instead of duplicating."""
        return f"recipe:{Path(self.source_file).stem}"

    def to_text(self) -> str:
        """The normalised text that gets embedded and shown to the LLM."""
        lines = [self.title, "", "Ingredients:"]
        lines += [f"- {item}" for item in self.ingredients]
        lines += ["", "Instructions:"]
        lines += [f"{n}. {step}" for n, step in enumerate(self.instructions, start=1)]
        return "\n".join(lines)


def parse_recipe(text: str, source_file: str) -> Recipe:
    """Parse one recipe file. Raises `RecipeParseError` naming the file if it does not fit."""
    title = ""
    sections: dict[str, list[str]] = {name: [] for name in _SECTIONS}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            name = header.group("name").strip().lower()
            if name not in _SECTIONS:
                raise RecipeParseError(f"{source_file}: unknown section {header.group('name')!r}")
            current = name
            continue
        if current is None:
            if not title:  # everything before the first header is the title
                title = line
            continue
        sections[current].append(_MARKER.sub("", line))

    if not title:
        raise RecipeParseError(f"{source_file}: no title found")
    for name in _SECTIONS:
        if not sections[name]:
            raise RecipeParseError(f"{source_file}: no {name} found")

    return Recipe(
        source_file=source_file,
        title=title,
        ingredients=sections["ingredients"],
        instructions=sections["instructions"],
    )


def recipe_to_document(recipe: Recipe) -> Document:
    """Wrap a recipe as the `Document` the indexing pipeline writes."""
    return Document(
        id=recipe.document_id,
        content=recipe.to_text(),
        meta={"title": recipe.title, "source_file": recipe.source_file},
    )


def load_recipes(recipes_dir: Path) -> list[Recipe]:
    """Parse every `*.txt` in `recipes_dir`, sorted by filename."""
    if not recipes_dir.is_dir():
        raise RecipeParseError(
            f"Recipes directory {recipes_dir} does not exist; run `unzip -o data.zip` first"
        )
    paths = sorted(recipes_dir.glob("*.txt"))
    if not paths:
        raise RecipeParseError(f"No .txt recipes found in {recipes_dir}")
    return [parse_recipe(path.read_text(encoding="utf-8"), path.name) for path in paths]
