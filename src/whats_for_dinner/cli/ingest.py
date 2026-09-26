"""`uv run ingest` — load the cookbook into the vector store. Idempotent."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from whats_for_dinner.core.config import Settings, configure_logging
from whats_for_dinner.core.exceptions import RecipeError
from whats_for_dinner.core.protocols import RecipeIndexer
from whats_for_dinner.domain.recipes import load_recipes, recipe_to_document
from whats_for_dinner.rag.adapters import build_indexer

logger = logging.getLogger(__name__)


def run_ingestion(recipes_dir: Path, indexer: RecipeIndexer) -> int:
    """Parse every recipe and write it to the store. Returns the number written."""
    recipes = load_recipes(recipes_dir)
    documents = [recipe_to_document(recipe) for recipe in recipes]
    written = indexer.index(documents)
    logger.info(
        "Ingestion finished: %d documents written from %d recipe files", written, len(recipes)
    )
    return written


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest recipes into the vector store.")
    parser.add_argument(
        "--recreate", action="store_true", help="drop and recreate the table (model changed)"
    )
    parser.add_argument("--recipes-dir", type=Path, default=None, help="defaults to RECIPES_DIR")
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)
    recipes_dir: Path = args.recipes_dir or settings.recipes_dir
    recreate: bool = args.recreate

    try:
        indexer = build_indexer(settings, recreate_table=recreate)
        logger.info("Ingesting into table %s from %s", indexer.table_name, recipes_dir)
        run_ingestion(recipes_dir, indexer)
    except RecipeError as exc:
        raise SystemExit(f"ingest failed: {exc}") from exc


if __name__ == "__main__":
    main()
