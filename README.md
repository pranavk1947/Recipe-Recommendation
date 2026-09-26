# What's for Dinner

POST the ingredients you have — text, a photo, or both — and get back one recipe from the provided
cookbook, adapted to what you actually have, in Markdown. FastAPI + Haystack 2.12 +
Postgres/pgvector + OpenAI; Swagger UI at `/docs` is the demo surface.

## 1. Steps to run

```bash
docker compose up -d db                 # Postgres with pgvector
unzip -o data.zip                       # 20 recipes + example photos
cp .env.example .env                    # then set OPENAI_API_KEY
uv sync
uv run ingest                           # embed the cookbook (idempotent)
uv run uvicorn whats_for_dinner.main:app --reload
```

Open http://localhost:8000/docs for Swagger UI — a form for the text and a file picker for the
photo — or use the requests in section 2.

Checks: `uv run ruff check . && uv run pyright && uv run pytest` — 21 offline tests, under a
second. `RUN_INTEGRATION=1 uv run pytest -m integration` needs Docker and a real key.

## 2. Usage examples

`POST /recommend_recipe` takes a form with `ingredients` (free text), `image` (optional photo), or
both. Three examples, with real responses (the Markdown shortened):

**1. Ingredients as text** — one recipe, plus the candidates it was chosen from and their scores:

```bash
curl -s -X POST localhost:8000/recommend_recipe \
  -F "ingredients=2 chicken breasts, a bag of spinach, feta, olive oil" | jq .
```

```json
{
  "recipe_markdown": "# Spinach and Feta Stuffed Chicken\n**Based on:** Spinach and Feta Stuffed Chicken (05.txt)\n\n## Why this recipe\n…\n## You have\n- 2 chicken breasts\n…\n## You still need\n- 1 tsp dried oregano (substitute with a pinch of any dried herbs you have…)\n…\n## Instructions\n1. Preheat your oven to 375°F (190°C).\n…",
  "matched_recipes": [
    {"title": "Spinach and Feta Stuffed Chicken", "source_file": "05.txt", "vector_score": 0.741,
     "keyword_score": 1.0, "rrf_score": 1.0, "matched_ingredients": ["chicken", "spinach", "feta", "olive oil"]},
    {"title": "Vegetable Frittata", "source_file": "17.txt", "vector_score": 0.523,
     "keyword_score": 0.5, "rrf_score": 0.968, "matched_ingredients": ["spinach", "olive oil"]},
    {"title": "Greek Salad", "source_file": "10.txt", "vector_score": 0.472,
     "keyword_score": 0.5, "rrf_score": 0.961, "matched_ingredients": ["feta", "olive oil"]}
  ],
  "detected_ingredients": null,
  "image_status": null,
  "llm_model": "gpt-4o"
}
```

**2. Gibberish text** — refused before any retrieval or recipe prompt:

```bash
curl -s -X POST localhost:8000/recommend_recipe -F "ingredients=asdkj qwpoe zzxv"
```

```
HTTP 400
{"error": "OffTopicInputError", "detail": "This service only recommends recipes from a list of ingredients - describe what you have in your kitchen"}
```

**3. A photo** — replace the path with any photo of ingredients (the cookbook ships examples in
`data/example_food_photos/`; this response is for `food1.webp`):

```bash
curl -s -X POST localhost:8000/recommend_recipe \
  -F "image=@/path/to/your-photo.jpg" | jq .
```

```json
{
  "recipe_markdown": "# Simple Guacamole\n**Based on:** Simple Guacamole (03.txt)\n\n## Why this recipe\nThis recipe is a great fit because it uses avocado and tomato, which you already have.\n…",
  "matched_recipes": [
    {"title": "Simple Guacamole", "source_file": "03.txt", "vector_score": 0.409,
     "keyword_score": 0.5, "rrf_score": 0.984, "matched_ingredients": ["avocado", "tomatoes"]},
    {"title": "Stuffed Bell Peppers", "source_file": "19.txt", "vector_score": 0.386,
     "keyword_score": 0.5, "rrf_score": 0.984, "matched_ingredients": ["tomatoes", "bell pepper"]},
    {"title": "Vegetarian Chili", "source_file": "12.txt", "vector_score": 0.356,
     "keyword_score": 0.5, "rrf_score": 0.961, "matched_ingredients": ["tomatoes", "bell pepper"]}
  ],
  "detected_ingredients": ["avocado", "broccoli", "tomatoes", "bell pepper"],
  "image_status": "ingredients_detected",
  "llm_model": "gpt-4o"
}
```

Text and a photo can be sent together (`-F "ingredients=…" -F "image=@…"`); ingredients from both
count. `uv run search "chicken, spinach, feta"` shows the ranking and scores without the answer.

## 3. Approach

RAG over a fixed 20-recipe cookbook. The LLM never invents a recipe; it adapts one the retriever
found, and the response carries the candidates so the choice is checkable without reading logs.
Built in four phases (section 5). Three decisions shape the rest (all of them, with reasons:
[PLAN.md](PLAN.md)):

- **Hybrid retrieval, not vector-only.** "What can I cook with these?" is partly a set-membership
  question — how many of my ingredients does this recipe use — which embeddings answer poorly and
  keyword search answers exactly. Both run; reciprocal rank fusion combines them.
- **Protocols at every boundary.** The service depends only on `core/` Protocols; Haystack
  pipelines, pgvector and the OpenAI SDK sit behind them. The service is testable with fakes. A
  provider seam exists, but only OpenAI is implemented, and vision and error translation are
  still OpenAI-specific: a second vendor would touch those too.
- **Errors carry their own status.** Every failure is a `RecipeError` subclass with `status_code`
  on the class, so one handler in `main.py` covers all of them and no HTTP detail leaks into the
  domain layer.

```
core/       config, exceptions, protocols, schemas   (depends on nothing of ours)
domain/     recipes, query, service                  (no HTTP, no pipelines)
providers/  registry, OpenAI provider, vision        storage/  repository (only pgvector importer)
rag/        pipelines, adapters, components, prompts
api/        routes, deps   cli/  ingest, search      main.py   composition root
```

Dependencies point inward only: `core ← domain, providers, storage ← rag ← api, cli`.

Two deliberate deviations from [CONVENTIONS.md](CONVENTIONS.md):

- **No SQLModel/Alembic:** Haystack's pgvector store creates and owns the one table, so a model
  and migrations would be a second source of truth fighting it; the only raw SQL is a read-only
  query kept in `storage/repository.py`.
- **Tests in `tests/`, not colocated:** the opt-in integration tests and the shared Protocol fakes
  in `conftest.py` span the whole app, so one tree mirroring `src/` holds them; files keep the
  `<module>_test.py` naming.

## 4. Assumptions

- **The cookbook is fixed:** the 20 recipes in `data.zip`, each a `.txt` with a title line and
  `Ingredients:` and `Instructions:` sections. A file that does not fit stops ingestion by name.
- **"Best matching" means** covering the most of your ingredients, then closest in meaning —
  hence hybrid retrieval. The answer is **one** recipe, adapted from a retrieved candidate and
  never invented; if none fits, the answer says so and still presents the closest.
- **Input is English**, like the cookbook. Quantities you give ("2 eggs") scale the recipe.
- **Text and photo are combined, not reconciled:** everything named in either counts as
  available, and negations ("no garlic") are not interpreted.
- **Off-topic text is refused** (400), unless a photo shows food to answer from. The check fails
  open: if it is down, the request is answered rather than refused.
- **OpenAI is the only provider** (the key is supplied): `gpt-4o` for answers and photos,
  `gpt-4o-mini` for the intent check, `text-embedding-3-small` for embeddings.
- **Postgres is the compose service;** the store creates its table on first use, so there are no
  migrations.
- **One-shot, stateless API:** no auth, no conversation history, no frontend. Photos in any
  Pillow-readable format, up to 10 MB.

## 5. Implementation, in four phases

The work is split into four phases, each building on the last and ending in a check you can run:

| Phase | What it delivers | Check |
|---|---|---|
| 1. Document ingestion | The cookbook parsed, embedded and stored in pgvector | `uv run ingest` → 20 documents written |
| 2. Query processing | Intent classification and ingredient extraction from text, photo input | off-topic text → 400 |
| 3. Hybrid retrieval | Vector and keyword search, fused by rank, with all three scores | `uv run search "…"` |
| 4. Response generation | GPT-4o adapts the best candidate into Markdown | `POST /recommend_recipe` |

### Phase 1 — Document ingestion into the vector DB

`uv run ingest` parses each `.txt` into title / ingredients / instructions, normalises it to one
text block, and writes it as a **single `Document`** — no chunking, because a recipe is 60–90 words
and splitting would separate the ingredient list from the method that uses it.

Embeddings are `text-embedding-3-small` (1536 dims) in `recipes_openai_1536`. The table name is
`<prefix>_<embedding_backend>_<dimension>`, so a second provider gets its own table and can never
read another model's vectors. Index names are per-table too — pgvector's defaults are schema-global
and collide.

Ids are deterministic (`recipe:01`) with `DuplicatePolicy.OVERWRITE`, so re-ingesting overwrites
rather than duplicates. There is no migration — the store issues its DDL lazily, which is why
startup calls `repository.count()`: one call creates the table, proves Postgres is reachable, and
reports whether ingestion has run.

### Phase 2 — Query processing with intent classification

**Text** is turned into keyword-search terms by the intent gate (below), which names each
ingredient without quantities, units or preparation:
`"500 grams minced beef, 1 packet of frozen peas"` → `["beef", "peas"]`. The embedder still
sees the raw text; terms drive keyword search only. Input is assumed to be English, like the
cookbook. This replaced a regex parser with hand-kept lists of units and fillers: any unit missing
from the list (`grams`, `tins`, `sprigs`…) stayed in the term, and because Postgres ANDs a term's
words, that ingredient then matched no recipe at all. If the gate is off or fails, there are no
terms: keyword search sits out and vector search alone ranks the recipes.

**Photos** go to `gpt-4o` vision via the OpenAI SDK directly — Haystack 2.12 has no image chat
content type — behind the `IngredientExtractor` Protocol, so the service is unaware. Any
Pillow-readable format is re-encoded to JPEG at `detail: "low"`. Vision replies with the same
JSON contract as the intent gate, `{"ingredients": [...]}` (empty for no food), so neither
extraction parses free text. The list goes **straight into the search terms**, never through the
gate: fed text and photo together, the gate dropped an unrelated photo as noise (10 of 10 runs for
"chicken, rice, soy sauce" plus a photo of fruit and yogurt). Vision and the gate are independent,
so they run concurrently. For the embedder and the recipe prompt both sources are concatenated,
user text first. Text and photo ingredients are
combined, not reconciled: "no garlic" in the text does not remove garlic seen in the photo.

`image_status` separates the three outcomes, which `detected_ingredients: null` alone cannot:
`null` (no photo sent), `"ingredients_detected"`, or `"no_food_detected"`.

An **intent gate** on a small model (`OPENAI_INTENT_MODEL`, default `gpt-4o-mini`) replies with
JSON — `{"is_ingredients": …, "ingredients": [...]}` — so one call both refuses off-topic text
(→ 400) and names the search terms. It reads only the typed text, and refuses only when no photo
showed food: gibberish text with a food photo is answered from the photo, and a photo alone skips
the call. It runs **in front of retrieval**, because retrieval needs those names: about 1–2 s
added to every answer, against about 3 s for generation. (It used to run concurrently with
retrieval, when it only returned a verdict.) Off-topic input now costs neither retrieval nor the
recipe prompt. It **fails open** — a guardrail outage or an unparseable reply
must not take the endpoint down, and vector search still answers without its terms — and
`INTENT_GATE_ENABLED=false` skips it entirely, answering everything with the closest match.

### Phase 3 — Hybrid retrieval

Two retrievers, fused with reciprocal rank fusion (top 3 of 8 candidates):

- **Vector** — the raw text embedded, cosine similarity, exact scan: perfect recall at 20 rows, and
  no `m`/`ef_search` to tune.
- **Keyword** — one SQL query, one `plainto_tsquery` **per ingredient** inside it, scored by
  coverage: the fraction of your ingredients the recipe contains.

Per-ingredient is the whole point. `plainto_tsquery` ANDs every word, so one tsquery for the full
sentence demands "bag" *and* "spinach" *and* "feta" in one recipe and matches nothing. The query
joins the recipes against one tsquery per term, counts the matches per recipe in SQL, and returns
the top 8 by that count, ties to the stronger text match (`ts_rank_cd`). Counting in the database
keeps coverage exact at any corpus size, uses the GIN index on `content`, and returns 8 rows
without their embeddings. It is plain SQL in `storage/repository.py`, because Haystack's keyword
retriever takes a single query string.

RRF because the two scores are not comparable: cosine similarity against a coverage fraction, so
only each recipe's *rank* in the two lists feeds the fused score. Each match reports all three, so
the order is explainable from the response alone:

```json
{"title": "Vegetable Frittata", "vector_score": 0.496, "keyword_score": 0.667, "rrf_score": 1.0}
```

`vector_score` is the cosine similarity, `keyword_score` the fraction of your ingredients the
recipe contains (`null` when the recipe was not among that retriever's 8 candidates — its
`matched_ingredients` is then empty), and `rrf_score` the fused score that sets the order: 1.0 =
ranked first by both, 0.5 = first by one only. The fusion is ours (`fuse_by_rank`), with the same formula as Haystack's `DocumentJoiner`,
because that joiner overwrites the per-retriever scores in place.

### Phase 4 — Response generation

The prompt passes ranked candidates and demands a fixed shape — `**Based on:** <candidate>`, then
*Why this recipe* / *You have* / *You still need* (with substitutions) / *Instructions*. It forbids
inventing a recipe outside the candidates, treats the user's text as data rather than instructions,
and scales quantities: 2 eggs against a recipe calling for 6 divides everything by three.

Three checks keep the output inspectable rather than trusted: a **grounding check** warns when no
retrieved title appears in the answer; **`matched_recipes`** ships title, source file, all three
scores and which of your ingredients were found, so the choice is auditable from the response
alone; and an **empty-index guard** returns 503 before the LLM call rather than a confident answer
over zero context. Every model call logs its model, tokens and duration:

```
step=intent.llm     model=gpt-4o-mini pipeline_duration=1.58s tokens=206+23
step=generation.llm model=gpt-4o      pipeline_duration=3.38s tokens=795+222
```
