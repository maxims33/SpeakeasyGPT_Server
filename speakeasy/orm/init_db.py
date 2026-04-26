"""
Backend-agnostic schema initializer.

Creates every table declared on ``Base.metadata`` against the configured
database (Postgres by default, SQLite when ``DATABASE_URL=sqlite:///sql.db``
or similar). Safe to re-run — ``create_all`` is a no-op for tables that
already exist.

Usage:
    python -m speakeasy.orm.init_db
"""

from sqlalchemy import inspect

from speakeasy.orm.db import engine
# Importing models registers them on ``Base.metadata`` so create_all picks
# them up. This must stay even though the names are not used directly.
from speakeasy.orm.models import Base  # noqa: F401
from speakeasy.orm import models  # noqa: F401


def init_db() -> None:
  """Create all tables on the configured database engine."""
  print(f"Initializing schema on: {engine.url.render_as_string(hide_password=True)}")
  Base.metadata.create_all(engine)
  inspector = inspect(engine)
  tables = sorted(inspector.get_table_names())
  print(f"Tables now present: {tables}")


if __name__ == "__main__":
  init_db()
