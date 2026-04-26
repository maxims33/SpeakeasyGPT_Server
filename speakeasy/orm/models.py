from typing import Optional

from sqlalchemy import Column, Integer, JSON, String, select
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, Session)

# Engine is owned by ``speakeasy.orm.db`` so that the database backend
# (Postgres by default, optionally SQLite) is configured in one place. We
# re-export it here for backwards compatibility with code that does
# ``from speakeasy.orm.models import engine``.
from speakeasy.orm.db import engine, get_session  # noqa: F401


class Base(DeclarativeBase):
  pass


class User(Base):
  __tablename__ = "AccountSettings"
  id: Mapped[int] = mapped_column(primary_key=True)
  username: Mapped[str] = mapped_column(String(30))
  password: Mapped[Optional[str]] = mapped_column(String(30))
  fullname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
  gender: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
  orientation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
  dateOfBirth: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

  # addresses: Mapped[List["Address"]] = relationship(
  #     back_populates="user", cascade="all, delete-orphan")

  def __repr__(self) -> str:
    return f"User(id={self.id!r}, username={self.username!r}, fullname={self.fullname!r}, password={self.password!r})"


class Address(Base):
  __tablename__ = "Addresses"
  id: Mapped[int] = mapped_column(primary_key=True)
  email_address: Mapped[str] = mapped_column(String(255))

  #  user_id: Mapped[int] = mapped_column(ForeignKey("User.id"))
  #  user: Mapped["User"] = relationship(back_populates="addresses")

  def __repr__(self) -> str:
    return f"Address(id={self.id!r}, email_address={self.email_address!r})"


class Recipe(Base):
  __tablename__ = 'recipes'
  id = Column(Integer, primary_key=True)
  # Widened from VARCHAR(50) — SQLite ignores length limits, but real recipe
  # names already exceed 50 chars and Postgres rejects them.
  name = Column(String(255), nullable=False)
  category = Column(String(100), nullable=False)
  ingredients = Column(JSON, nullable=False)
  instructions = Column(JSON, nullable=False)
  image_file = Column(String(255), nullable=True)

  def __repr__(self):
    return f"Recipe(name='{self.name}')"


# ----------------------------------------


def insert(obj, session):
  session.add_all([obj])
  session.commit()


def find_user(username, session):
  stmt = select(User).where(User.username.in_([username]))

  for user in session.scalars(stmt):
    return user

  return None


def find_user_with_password(username, password, session):
  stmt = select(User).where(User.username == username).where(
      User.password == password)

  for user in session.scalars(stmt):
    return user

  return None


def update(obj, session):
  session.commit()


def find_recipes(category: str = ""):
  with Session(engine) as session:
    stmt = select(Recipe).where(Recipe.category == category)
    return session.scalars(stmt).all()
