"""
#TODO Additional params to potentially expose in future
#TODO Consider using dotenv to simplyfy handling of envrionment variables
"""

import os

# Default SQLAlchemy URL used when no Postgres credentials are configured.
DEFAULT_SQLITE_URL = "sqlite:///sql.db"


def _resolve_database_url():
    """Resolve the SQLAlchemy URL for the relational database.

    Resolution order:
      1. ``DATABASE_URL`` environment variable (highest priority — supports any
         SQLAlchemy URL, e.g. ``sqlite:///sql.db`` to fall back to the legacy
         backend, or a custom Postgres host).
      2. Standard ``PG*`` environment variables (``PGHOST``, ``PGPORT``,
         ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE``) — built into a
         ``postgresql+psycopg://`` URL when all required parts are present.
      3. ``DEFAULT_SQLITE_URL`` (legacy local SQLite file) as a final fallback.
    """
    raw = os.environ.get("DATABASE_URL")
    if raw:
        # Normalize legacy ``postgres://`` and ``postgresql://`` URLs to use
        # the modern psycopg (v3) driver explicitly.
        if raw.startswith("postgres://"):
            raw = "postgresql+psycopg://" + raw[len("postgres://") :]
        elif raw.startswith("postgresql://"):
            raw = "postgresql+psycopg://" + raw[len("postgresql://") :]
        return raw

    host = os.environ.get("PGHOST")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    database = os.environ.get("PGDATABASE")
    if host and user and database:
        from urllib.parse import quote_plus

        port = os.environ.get("PGPORT", "5432")
        userinfo = quote_plus(user)
        if password:
            userinfo += ":" + quote_plus(password)
        return f"postgresql+psycopg://{userinfo}@{host}:{port}/{database}"

    return DEFAULT_SQLITE_URL


def default_variables():
    """Setting defaults for all environment variables"""
    return {
        "debug": False,
        "docs_persist_directory": "./db/",
        "images_persist_directory": "./db2/",
        "factory_type": "GOOGLE",
        "image_directory": "./generated_images/",
        "google_llm_api_timeout": 30,
        "auth": "firebase",  # dummy / firebase
        "database_url": _resolve_database_url(),
        #'local_model_name': 'google/flan-t5-large',
        #'device_id': 'cpu',
        #'max_iterations': 1,
        #'max_length': 512,
        #'embedding_model_name': 'intfloat/e5-large-v2',
        #'embedding_device_id': 'cuda',
        #'source_chunks': 1,
        #'text_split_size': 1000,
        #'text_split_overlap': 200,
    }


def parse_environment_variables():
    """Setting the configured values, if defined"""
    print("Loading environment variables")
    env_variables = default_variables()

    if (
        os.environ.get("ENABLE_DEBUG") == "True"
        or os.environ.get("ENABLE_DEBUG") == "true"
    ):
        env_variables["debug"] = True

    if not os.environ.get("DOCS_PERSIST_DIRECTORY") is None:
        env_variables["docs_persist_directory"] = os.environ.get(
            "DOCS_PERSIST_DIRECTORY"
        )

    if not os.environ.get("IMAGES_PERSIST_DIRECTORY") is None:
        env_variables["images_persist_directory"] = os.environ.get(
            "IMAGES_PERSIST_DIRECTORY"
        )

    if not os.environ.get("IMAGE_DIRECTORY") is None:
        env_variables["image_directory"] = os.environ.get("IMAGE_DIRECTORY")

    if not os.environ.get("FACTORY_TYPE") is None:
        env_variables["factory_type"] = os.environ.get("FACTORY_TYPE")

    if not os.environ.get("LOCAL_MODEL_NAME") is None:
        env_variables["local_model_name"] = os.environ.get("LOCAL_MODEL_NAME")

    if not os.environ.get("EMBEDDING_MODEL_NAME") is None:
        env_variables["embedding_model_name"] = os.environ.get("EMBEDDING_MODEL_NAME")

    if not os.environ.get("GOOGLE_LLM_API_TIMEOUT") is None:
        env_variables["google_llm_api_timeout"] = int(
            os.environ.get("GOOGLE_LLM_API_TIMEOUT")
        )

    return env_variables


# Collect environment variables or defaults
env_config = parse_environment_variables()
