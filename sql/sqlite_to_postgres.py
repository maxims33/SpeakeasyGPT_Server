"""
One-off data migration: copy rows from the legacy SQLite file into the
configured Postgres (or any non-sqlite) database.

The script:
  * Opens a *source* engine pointing at ``sqlite:///sql.db`` (override with
    ``--source`` or ``SOURCE_DATABASE_URL``).
  * Opens a *target* engine using ``env_params.env_config['database_url']``
    (override with ``--target`` or ``TARGET_DATABASE_URL``). Refuses to run
    if the target also resolves to SQLite, unless ``--allow-sqlite-target``
    is passed.
  * Ensures ``Base.metadata`` is created on the target.
  * Copies all rows for ``User`` (``AccountSettings``), ``Address``
    (``Addresses``) and ``Recipe`` (``recipes``) inside a single
    transaction.
  * Is idempotent: rows whose primary key already exists on the target are
    skipped, so re-running is safe.
  * Resets each table's primary-key sequence on Postgres so future inserts
    don't collide with copied IDs.
  * Leaves the source SQLite file completely untouched (read-only access).

Known data quirk — double-encoded JSON columns:
  Rows in the legacy ``sql.db`` were written with ``json.dumps(value)`` into
  what was then a plain TEXT column. Since the ORM now declares those
  columns as ``Column(JSON)``, SQLAlchemy's JSON deserializer runs on read
  and peels off one layer of encoding -- but the original write put two
  layers in, so what we hand to the target side is still a JSON-encoded
  string (e.g. ``'["a","b"]'``) rather than the decoded list ``["a","b"]``.
  We deliberately copy values verbatim here, so on the target Postgres these
  rows land as ``json_typeof = 'string'`` instead of ``'array'``. Cleanly
  inserted rows (added after the migration) land as proper ``'array'``
  values. Readers must therefore tolerate both shapes -- see
  ``speakeasy/nutrition_content.py`` for the defensive parse.

Usage:
    python -m scripts.sqlite_to_postgres
    python -m scripts.sqlite_to_postgres --source sqlite:///sql.db --target $DATABASE_URL
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Type

from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

# Make sure the project root is on sys.path when invoked via ``python scripts/...``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_params import env_config  # noqa: E402
from speakeasy.orm.db import build_engine  # noqa: E402
from speakeasy.orm.models import Address, Base, Recipe, User  # noqa: E402

MIGRATED_MODELS: List[Type[Base]] = [User, Address, Recipe]


def _row_to_dict(row: Base) -> dict:
  """Return a dict of column -> value for the given ORM row.

  Values are passed through untouched. Note that for ``Column(JSON)`` fields
  the legacy SQLite rows are double-encoded (see "Known data quirk" in the
  module docstring), so what we read here for those fields is a JSON string,
  not a decoded list/dict. We intentionally carry it across as-is rather
  than try to re-decode and risk corrupting other rows that may already be
  single-encoded.
  """
  return {col.name: getattr(row, col.name) for col in row.__table__.columns}


def _copy_table(model: Type[Base], src: Session, dst: Session) -> tuple[int, int]:
  """Copy rows of ``model`` from ``src`` to ``dst``. Returns (copied, skipped)."""
  table = model.__table__
  src_inspector = inspect(src.get_bind())
  if table.name not in src_inspector.get_table_names():
    print(f"  - {table.name}: not present on source, skipping")
    return (0, 0)

  pk_cols = [c.name for c in table.primary_key.columns]
  if not pk_cols:
    raise RuntimeError(f"Table {table.name} has no primary key, cannot migrate safely")
  pk_col = pk_cols[0]

  existing_ids = {row[0] for row in dst.execute(select(getattr(model, pk_col))).all()}
  rows: Iterable[Base] = src.execute(select(model)).scalars().all()

  copied = 0
  skipped = 0
  for row in rows:
    pk_val = getattr(row, pk_col)
    if pk_val in existing_ids:
      skipped += 1
      continue
    dst.execute(table.insert().values(**_row_to_dict(row)))
    copied += 1
  return (copied, skipped)


def _reset_postgres_sequences(engine: Engine) -> None:
  """Bump each table's PK sequence past the max copied id (Postgres only)."""
  if engine.dialect.name != "postgresql":
    return
  with engine.begin() as conn:
    for model in MIGRATED_MODELS:
      table = model.__table__
      pk_cols = [c.name for c in table.primary_key.columns]
      if len(pk_cols) != 1:
        continue
      pk_col = pk_cols[0]
      # ``pg_get_serial_sequence`` requires the table identifier exactly as
      # Postgres stores it. Our tables use mixed-case names, so we must pass
      # the fully-quoted form (``"AccountSettings"``) — otherwise Postgres
      # folds it to lowercase and the lookup fails.
      seq = conn.execute(
          text("SELECT pg_get_serial_sequence(:t, :c)"),
          {"t": f'"{table.name}"', "c": pk_col},
      ).scalar()
      if not seq:
        continue
      max_id = conn.execute(
          text(f'SELECT COALESCE(MAX("{pk_col}"), 0) FROM "{table.name}"')
      ).scalar() or 0
      conn.execute(
          text("SELECT setval(:s, :v, true)"),
          {"s": seq, "v": max(max_id, 1)},
      )
      print(f"  - sequence {seq} set to {max(max_id, 1)}")


def migrate(source_url: str, target_url: str) -> None:
  if source_url == target_url:
    raise SystemExit(f"Source and target URLs are identical: {source_url}")

  src_engine = build_engine(source_url)
  dst_engine = build_engine(target_url)

  print(f"Source: {src_engine.url.render_as_string(hide_password=True)}")
  print(f"Target: {dst_engine.url.render_as_string(hide_password=True)}")

  print("Ensuring target schema exists ...")
  Base.metadata.create_all(dst_engine)

  with Session(src_engine) as src, Session(dst_engine) as dst:
    print("Copying tables ...")
    summary = []
    for model in MIGRATED_MODELS:
      copied, skipped = _copy_table(model, src, dst)
      summary.append((model.__tablename__, copied, skipped))
      print(f"  - {model.__tablename__}: copied={copied} skipped={skipped}")
    dst.commit()

  _reset_postgres_sequences(dst_engine)

  print("\nMigration summary:")
  for name, copied, skipped in summary:
    print(f"  {name:20s} copied={copied:>4d}  skipped(existing)={skipped:>4d}")
  print("Done.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--source",
      default=os.environ.get("SOURCE_DATABASE_URL", "sqlite:///sql.db"),
      help="SQLAlchemy URL for the source DB (default: sqlite:///sql.db).",
  )
  parser.add_argument(
      "--target",
      default=os.environ.get("TARGET_DATABASE_URL", env_config["database_url"]),
      help="SQLAlchemy URL for the target DB (default: env_config.database_url).",
  )
  parser.add_argument(
      "--allow-sqlite-target",
      action="store_true",
      help="Permit copying *into* a SQLite database (off by default).",
  )
  args = parser.parse_args()

  if args.target.startswith("sqlite") and not args.allow_sqlite_target:
    raise SystemExit(
        "Refusing to migrate into a SQLite target. Set DATABASE_URL to your "
        "Postgres URL or pass --allow-sqlite-target.")

  migrate(args.source, args.target)


if __name__ == "__main__":
  main()
