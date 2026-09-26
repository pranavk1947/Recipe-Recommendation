"""The only module that knows about pgvector and psycopg."""

import psycopg
from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from psycopg import sql
from psycopg.rows import dict_row

from whats_for_dinner.core.config import Settings
from whats_for_dinner.core.exceptions import RecipeStoreUnavailableError
from whats_for_dinner.core.protocols import LLMProvider

# Shared by the store's full-text index and the coverage query: if they differ, Postgres cannot
# use the index.
_TEXT_SEARCH_LANGUAGE = "english"

# One `plainto_tsquery` per term, because a single query over all terms would AND every word and
# match nothing. Counting matched terms in SQL keeps coverage exact at any corpus size and returns
# only `top_k` rows, without their embeddings. Equal coverage goes to the stronger text match.
# It also names the terms each recipe matched, so `matched_ingredients` and `keyword_score` come
# from the same Postgres match and cannot disagree.
_COVERAGE_QUERY = """
WITH q AS (
    SELECT t.term, t.position, plainto_tsquery({language}, t.term) AS query
    FROM unnest(%(terms)s::text[]) WITH ORDINALITY AS t(term, position)
)
SELECT d.id, d.content, d.meta, array_agg(q.term ORDER BY q.position) AS matched_terms
FROM {table} AS d
JOIN q ON to_tsvector({language}, d.content) @@ q.query
GROUP BY d.id
ORDER BY count(*) DESC, sum(ts_rank_cd(to_tsvector({language}, d.content), q.query)) DESC, d.id
LIMIT %(top_k)s
"""


class RecipeRepository:
    """Owns the recipe table for one embedding backend."""

    def __init__(
        self, settings: Settings, provider: LLMProvider, *, recreate_table: bool = False
    ) -> None:
        # One table per backend+dimension: a second provider can never read another
        # model's vectors or hit a dimension mismatch.
        self.table_name = (
            f"{settings.recipes_table_prefix}_"
            f"{provider.embedding_backend}_{provider.embedding_dimension}"
        )
        self._dsn = settings.postgres_dsn
        self._connection: psycopg.AsyncConnection | None = None
        self._store = PgvectorDocumentStore(
            connection_string=Secret.from_token(settings.postgres_dsn),
            table_name=self.table_name,
            embedding_dimension=provider.embedding_dimension,
            vector_function="cosine_similarity",
            # Exact scan: at 20 recipes a full scan is instant and has perfect recall.
            search_strategy="exact_nearest_neighbor",
            recreate_table=recreate_table,
            language=_TEXT_SEARCH_LANGUAGE,
            # Defaults are schema-global and would collide on a second table.
            keyword_index_name=f"{self.table_name}_keyword_idx",
            # Switch to approximate HNSW once the corpus is large (~10k+ rows). Haystack only
            # builds this index when search_strategy="hnsw".
            # search_strategy="hnsw",
            # hnsw_index_name=f"{self.table_name}_hnsw_idx",
        )

    @property
    def document_store(self) -> PgvectorDocumentStore:
        """The underlying pgvector store for this backend's table."""
        return self._store

    def embedding_retriever(self, top_k: int) -> PgvectorEmbeddingRetriever:
        """Cosine-similarity vector search returning the `top_k` nearest recipes."""
        return PgvectorEmbeddingRetriever(document_store=self._store, top_k=top_k)

    async def coverage_search(self, terms: list[str], top_k: int) -> list[Document]:
        """The `top_k` recipes mentioning the most `terms`, scored by the fraction mentioned.

        Each document's `meta["matched_terms"]` lists the terms it matched, in the order given.
        """
        if not terms:
            return []
        query = sql.SQL(_COVERAGE_QUERY).format(
            language=sql.Literal(_TEXT_SEARCH_LANGUAGE), table=sql.Identifier(self.table_name)
        )
        try:
            connection = await self._open_connection()
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(query, {"terms": terms, "top_k": top_k})
                rows = await cursor.fetchall()
        except psycopg.Error as exc:
            raise RecipeStoreUnavailableError(f"Recipe store failed: {exc}") from exc
        return [
            Document(
                id=str(row["id"]),
                content=str(row["content"]),
                meta={**row["meta"], "matched_terms": list(row["matched_terms"])},
                score=len(row["matched_terms"]) / len(terms),
            )
            for row in rows
        ]

    async def aclose(self) -> None:
        """Close the coverage-query connection; called when the app shuts down."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def _open_connection(self) -> psycopg.AsyncConnection:
        """One connection, opened on first use and reused.

        Connecting costs about twice the query.
        """
        connection = self._connection
        if connection is None or connection.closed or connection.broken:
            connection = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
            self._connection = connection
        return connection

    def document_writer(self) -> DocumentWriter:
        """Writes embedded recipes, overwriting any with the same id on re-ingest."""
        return DocumentWriter(document_store=self._store, policy=DuplicatePolicy.OVERWRITE)

    async def count(self) -> int:
        """How many recipes are indexed. The store connects lazily, so this creates the table."""
        try:
            return await self._store.count_documents_async()
        except (DocumentStoreError, psycopg.Error) as exc:
            raise RecipeStoreUnavailableError(f"Recipe store failed: {exc}") from exc
