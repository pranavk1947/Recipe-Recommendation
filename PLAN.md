# Plan — What's for Dinner

Recommend one recipe from the provided cookbook for the ingredients a user has, as Markdown.
Inputs: [CHALLENGE.md](CHALLENGE.md) (what to build) and [CONVENTIONS.md](CONVENTIONS.md) (how).

## 1. Requirements

| # | Requirement | Source |
|---|---|---|
| R1 | `POST /recommend_recipe` on FastAPI, free-text ingredients, Markdown answer | Challenge §1 |
| R2 | GPT-4o generates the answer | Challenge §1 |
| R3 | Bonus: accept a photo of ingredients alongside the text | Challenge §1 |
| R4 | Recipes come from `data.zip`; Postgres finds the best matches | Challenge §2 |
| R5 | Ingestion at startup or as a separate command | Challenge §2 |
| R6 | RAG, preferably Haystack 2.x | Challenge §2 |
| R7 | uv, `src` layout; README with run steps, architecture, assumptions, examples | Challenge §3–4 |
| R8 | PoC in ≤ 5 h: explainable and defensible, not over-engineered | Challenge §5 |
| C1 | Typed Python 3.12, Protocols over concretions, thin API, top-down service, early returns | Conventions |
| C2 | Pydantic models and Settings, domain exceptions handled at the API layer | Conventions |
| C3 | ruff + pyright clean, pytest with `<module>_test.py`, `anyio` for async | Conventions |

## 2. Decisions

Each decision names the requirement it serves.

**Data and ingestion**
- **D1 — One document per recipe, no chunking** (R4, R6). A recipe is 60–90 words; splitting would
  separate the ingredient list from the steps that use it. Parse each `.txt` into title,
  ingredients and instructions, then embed one normalised text block.
- **D2 — Ingestion is a CLI command, `uv run ingest`** (R5). Embedding costs API calls and
  happens rarely; the server must not repeat it on every start. The server only warns when the
  index is empty.
- **D3 — Deterministic ids (`recipe:<file>`) with overwrite** (R5). Re-running ingest replaces
  rows instead of duplicating them. Overwrite never deletes, so `--recreate` rebuilds from empty
  after recipes are removed or the model changes at the same dimension (D4 covers the rest).
- **D4 — Haystack's pgvector store, one table per embedding backend and dimension**
  (R4, R6). `recipes_<backend>_<dim>`: vectors from different models never mix, and a second
  provider needs no migration. Index names are per table because pgvector's defaults are global.

**Query understanding**
- **D5 — One small-model LLM call (the intent gate) both classifies and extracts** (R1, R8).
  It replies with JSON `{"is_ingredients", "ingredients"}`: off-topic text gets a 400 before any
  retrieval, and the plain ingredient names ("500 grams minced beef" → "beef") become the keyword
  terms. No hand-kept lists of units and fillers. Runs on `gpt-4o-mini` because it sits in front
  of every request.
- **D6 — The gate fails open** (R1). An outage or unparseable reply must not take the endpoint
  down: the request proceeds with no terms and vector search alone ranks.
- **D7 — Photos go through gpt-4o vision; their ingredients skip the gate** (R3). Vision replies
  with the gate's JSON contract (`{"ingredients": [...]}`), so the list goes straight into the
  terms; the gate reads only the typed text, concurrently with vision, and refuses only when no
  photo showed food. `image_status` separates "no photo" from "no food".
  Vision uses the OpenAI SDK directly because Haystack 2.12 has no image chat content.

**Retrieval**
- **D8 — Hybrid retrieval, fused by rank** (R4, R6). "What can I cook with these?" is partly set
  membership, which embeddings answer poorly. Vector search (cosine, exact scan) and keyword
  search each return 8 candidates; reciprocal rank fusion (k = 61) keeps the top 3. Keyword search
  is one SQL query with a full-text match per ingredient, counting matches per recipe in the
  database: coverage stays exact at any corpus size, and only 8 rows come back, no embeddings.
  The same query names the matched terms, so `matched_ingredients` agrees with `keyword_score`.
- **D9 — Report every score** (R1, R8). Each match carries `vector_score`, `keyword_score` and
  `rrf_score`, so the ranking is explainable from the response. Fusion is done in our code
  (`fuse_by_rank`) because Haystack's joiner overwrites the per-retriever scores.

**Generation**
- **D10 — The LLM adapts a retrieved recipe, never invents one** (R2, R6). The prompt passes the
  ranked candidates, fixes the Markdown sections, scales quantities to what the user has, and
  treats user text as data. A grounding check logs when no candidate title appears in the answer.

**Structure**
- **D11 — Protocols at every boundary** (C1). The service depends only on `core/` Protocols;
  Haystack pipelines, pgvector and the OpenAI SDK sit behind adapters. The service is testable
  with fakes. A provider seam exists (a registry and the `LLMProvider` Protocol), but only OpenAI
  is implemented, and vision, error translation and the settings are still OpenAI-specific: a
  second vendor would touch those too. Accepted leak: Haystack's small `Document` type is used in
  `core/protocols.py` and `domain/recipes.py` rather than mirrored by a type of our own.
- **D12 — Errors carry their HTTP status** (C2). Every failure subclasses `RecipeError` with a
  `status_code`, so one handler in `main.py` maps all of them.
- **D13 — Every model call logs its model, tokens and time** (R8) — the pipeline's duration for
  pipeline steps, the call's own for vision. Latency and token use are visible without extra tooling.

## 3. Phases

Built in four phases, each on top of the last and ending in a check that runs. D11–D13 are laid
down in phase 1 and hold throughout.

| Phase | Delivers | Decisions | Check |
|---|---|---|---|
| 1. Document ingestion | Cookbook parsed, embedded and stored in pgvector | D1–D4 | `uv run ingest` → 20 documents |
| 2. Query processing | Intent classification, ingredient extraction, photo input | D5–D7 | off-topic text → 400 |
| 3. Hybrid retrieval | Vector + keyword search, rank fusion, all three scores | D8–D9 | `uv run search "…"` |
| 4. Response generation | GPT-4o adapts the best candidate into Markdown | D10 | `POST /recommend_recipe` |

## 4. Architecture

Dependencies point inward: `core ← domain, providers, storage ← rag ← api, cli`.

```
src/whats_for_dinner/
├── main.py              composition root: builds components once, wires FastAPI, error handler
├── api/
│   ├── routes.py        endpoints only: /recommend_recipe, /health
│   └── dependencies.py  startup components as FastAPI dependencies; form check, image upload
├── cli/
│   ├── ingest.py        uv run ingest [--recreate]
│   └── search.py        uv run search "…": retrieval with scores, no answer LLM
├── core/                depends on nothing of ours
│   ├── config.py        Pydantic Settings from env / .env
│   ├── exceptions.py    RecipeError hierarchy with status codes
│   ├── protocols.py     IntentClassifier, RecipeRetriever, RecipeGenerator, RecipeIndexer, …
│   └── schemas.py       request/response and internal Pydantic models
├── domain/              no HTTP, no pipelines
│   ├── recipes.py       .txt → Recipe → Document
│   ├── query.py         ingredient names → search terms
│   └── service.py       guard → vision ∥ gate → retrieve → generate → respond
├── providers/
│   ├── __init__.py      provider registry (LLM_PROVIDER)
│   ├── openai_provider.py  embedders and chat generators for OpenAI
│   └── vision.py        photo → ingredient list
├── rag/
│   ├── pipelines.py     Haystack wiring only: indexing, retrieval, intent, generation
│   ├── components.py    keyword-retriever component, rank fusion
│   ├── adapters.py      Protocol implementations over pipelines; error translation
│   └── prompts.py       intent and recipe prompts
└── storage/repository.py  the only module that imports pgvector / psycopg; the coverage query
tests/                   mirrors src/; integration/ is opt-in (RUN_INTEGRATION=1)
```

## 5. Request flow

1. Guard: no text and no photo → 400.
2. Concurrently: photo → vision → ingredient list (empty = no food); typed text → gate.
3. Nothing usable from either → 400.
4. Refuse (400) only if the gate judged the text off-topic *and* no photo showed food.
5. Terms = typed names + photo names. Retrieve (hybrid, fused) → none indexed → 503.
6. Generate Markdown from the candidates; check grounding.
7. Respond: Markdown, matches with all three scores, detected ingredients, `image_status`.

## 6. Cases to Handle

Every combination of text and photo, and what the request flow above does with it.

| # | Text | Photo | Result | Search terms from |
|---|---|---|---|---|
| 1 | — | — | 400 `NoIngredientsProvidedError` | — |
| 2 | ingredients | — | 200 | gate |
| 3 | off-topic or gibberish | — | 400 `OffTopicInputError`, no retrieval | — |
| 4 | — | food | 200, gate not called | photo |
| 5 | — | no food | 400 `NoIngredientsProvidedError` (nothing to answer from) | — |
| 6 | ingredients | food, matching | 200 | gate + photo |
| 7 | ingredients | food, unrelated | 200, both sets count as available | gate + photo |
| 8 | off-topic or gibberish | food | 200, answered from the photo | photo |
| 9 | ingredients | no food | 200, `no_food_detected` | gate |
| 10 | off-topic or gibberish | no food | 400 `OffTopicInputError` | — |
| 11 | "no garlic" | shows garlic | 200, garlic counts (negation not interpreted) | gate + photo |

Failure and edge cases:

| Case | Result |
|---|---|
| Empty image field (Swagger UI sends `image=""`) | Treated as no photo |
| Unreadable image, or larger than 10 MB | 400 `InvalidImageError` |
| JSON body instead of a form | 415 `UnsupportedMediaTypeError` |
| Photo sent, provider has no vision | 501 `ImageInputUnavailableError` |
| Gate down or reply not valid JSON | Fails open: answered, no typed terms, vector search ranks |
| Gate disabled (`INTENT_GATE_ENABLED=false`) | Nothing refused; closest match returned |
| An OpenAI call fails (vision, embedding or answer) | 502 |
| OpenAI rejects the API key | 500 `ProviderConfigurationError`, naming `OPENAI_API_KEY` |
| Recipe answer cut off at the token limit | 502 `RecommendationGenerationError`, never half a recipe |
| Nothing ingested yet | 503 `RecipeIndexEmptyError`, before the answer LLM runs |
| Postgres unreachable | 503 `RecipeStoreUnavailableError`; the server refuses to start |

## 7. Testing

Only the critical flows, one test each, all offline with fakes at the Protocol seams:
ingestion (recipe parsing, stable ids, the real cookbook parses); query processing (off-topic
never reaches retrieval, gate names drive keyword search, photo terms survive unrelated text,
gibberish plus a food photo is answered, the gate fails open); retrieval (rank fusion and its
tie-break); generation guards (empty index → 503, a truncated answer is an error, a rejected
key is a configuration error, errors keep their status); the HTTP form and image upload. Three opt-in end-to-end tests run against real Postgres and OpenAI: a real
request, an off-topic refusal, and the coverage query (exact counts, no embeddings returned).

## 8. Deviations from the conventions

- **No SQLModel or Alembic.** Haystack's pgvector store owns its schema and creates it lazily;
  a single table needs no migrations. Startup calls `count()` to create it and prove Postgres is up.
- **Tests live in `tests/`, mirroring `src/`,** named `<module>_test.py`, so the package ships
  without them.
- **Log values go in the message, not only `extra=`,** so they show in plain console output.

## 9. Out of scope

English input only; negations ("no garlic") are not interpreted; text and photo ingredients are
combined, not reconciled; no conversation or follow-ups; no frontend — Swagger UI at `/docs` is
the demo surface; approximate vector indexing (HNSW) waits until the corpus of recipes reaches a
suitable scale.
