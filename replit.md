# SpeakeasyGPT Server — Replit Notes

Flask-based REST server that wraps multiple LLM providers (LangChain, Google Gen AI, HuggingFace), Chroma vector stores, and a relational database for user accounts and recipes.

## Persistence layer

| Concern | Where |
|---|---|
| SQLAlchemy engine + session factory | `speakeasy/orm/db.py` |
| Models (`User`, `Address`, `Recipe`) and query helpers | `speakeasy/orm/models.py` |
| Schema initializer (Postgres or SQLite) | `speakeasy/orm/init_db.py` (`python -m speakeasy.orm.init_db`) |
| One-off SQLite → Postgres data migration | `scripts/sqlite_to_postgres.py` (`python -m scripts.sqlite_to_postgres`) |
| Backend selection | `env_params._resolve_database_url()` |

The relational database backend is **Postgres by default** and is configured through a single environment variable:

* Set `DATABASE_URL` to any SQLAlchemy URL to override.
* If `DATABASE_URL` is unset, the app falls back to standard `PG*` env vars (`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`) — what Replit's managed Postgres exposes.
* If neither is present, the legacy `sqlite:///sql.db` file is used.

Legacy `postgres://` / `postgresql://` URLs are rewritten to use the modern `psycopg` (v3) driver automatically.

The original `sql.db` file is preserved on disk so SQLite remains a one-line fallback (`DATABASE_URL=sqlite:///sql.db`).

### Tables

| Model | Table | Notes |
|---|---|---|
| `User` | `AccountSettings` | Account profile fields. `password` is plain text (legacy). |
| `Address` | `Addresses` | Reserved — no rows in legacy data. |
| `Recipe` | `recipes` | `ingredients` / `instructions` are JSON columns. |

`Recipe.name` was widened from `VARCHAR(50)` to `VARCHAR(255)` and `Recipe.category` from `VARCHAR(50)` to `VARCHAR(100)` because real recipe names exceed 50 chars (SQLite ignored those length limits, Postgres enforces them).

### Vector stores

ChromaDB persistence directories are unrelated to the relational DB and are still configured via `DOCS_PERSIST_DIRECTORY` / `IMAGES_PERSIST_DIRECTORY` (defaults: `./db/`, `./db2/`).

## Useful commands

```bash
# Initialize / verify schema on the configured backend
python -m speakeasy.orm.init_db

# Copy data from sql.db into Postgres (idempotent, leaves sql.db untouched)
python -m scripts.sqlite_to_postgres

# Run with SQLite instead of Postgres
DATABASE_URL=sqlite:///sql.db ./scripts/runserver.sh
```

## Notes

* `sql/sql_snippets.py` is dev scratch using raw `sqlite3` and is not imported by the running server. Don't add new code there — use the SQLAlchemy models in `speakeasy/orm/models.py`.
* `sql/init_sqlite.py` is kept for backwards compatibility but now just calls `init_db()`.
