"""
Legacy SQLite-only initializer. Kept for backwards compatibility — the
project now provisions schema in a backend-agnostic way via SQLAlchemy.

Prefer:

    python -m speakeasy.orm.init_db

which will create the schema on whatever database ``DATABASE_URL`` is
pointing at (Postgres by default, or SQLite when explicitly configured).
"""

from speakeasy.orm.init_db import init_db


if __name__ == "__main__":
  print("[deprecated] sql/init_sqlite.py now delegates to "
        "`python -m speakeasy.orm.init_db`.")
  init_db()
