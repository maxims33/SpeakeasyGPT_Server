"""
Centralized database engine/session factory.

Reads the configured backend URL from ``env_params.env_config['database_url']``
so the same code works against Postgres (default) or SQLite by changing a
single environment variable (``DATABASE_URL``). Engine arguments are tuned per
backend (e.g. SQLite needs ``check_same_thread=False`` when shared across
threads).
"""

from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from env_params import env_config


def _engine_kwargs(url: str) -> Dict[str, Any]:
  """Return backend-appropriate kwargs for ``create_engine``."""
  kwargs: Dict[str, Any] = {"echo": bool(env_config.get("debug", False))}
  if url.startswith("sqlite"):
    # Allow SQLite connections to be used across Flask's request threads.
    kwargs["connect_args"] = {"check_same_thread": False}
  else:
    # Reasonable defaults for Postgres / other server-based backends.
    kwargs["pool_pre_ping"] = True
    kwargs["pool_recycle"] = 1800
  return kwargs


def build_engine(url: str | None = None) -> Engine:
  """Build a SQLAlchemy engine for the given (or configured) URL."""
  resolved = url or env_config["database_url"]
  return create_engine(resolved, **_engine_kwargs(resolved))


# Module-level engine + session factory used by the app.
engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def get_session() -> Session:
  """Return a new SQLAlchemy ``Session`` bound to the configured engine."""
  return SessionLocal()
