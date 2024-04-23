from typing import List, Optional
from sqlalchemy import ForeignKey, String, create_engine, select
from sqlalchemy.engine.result import null_result
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, Session)


class Base(DeclarativeBase):
  pass


class User(Base):
  __tablename__ = "AccountSettings"
  id: Mapped[int] = mapped_column(primary_key=True)
  username: Mapped[str] = mapped_column(String(30))
  password: Mapped[Optional[str]] = mapped_column(String(30))
  fullname: Mapped[Optional[str]]
  gender: Mapped[Optional[str]]
  orientation: Mapped[Optional[str]]
  dateOfBirth: Mapped[Optional[str]]

  # addresses: Mapped[List["Address"]] = relationship(
  #     back_populates="user", cascade="all, delete-orphan")

  def __repr__(self) -> str:
    return f"User(id={self.id!r}, username={self.username!r}, fullname={self.fullname!r}, password={self.password!r})"


class Address(Base):
  __tablename__ = "Addresses"
  id: Mapped[int] = mapped_column(primary_key=True)
  email_address: Mapped[str]

  #  user_id: Mapped[int] = mapped_column(ForeignKey("User.id"))
  #  user: Mapped["User"] = relationship(back_populates="addresses")

  def __repr__(self) -> str:
    return f"Address(id={self.id!r}, email_address={self.email_address!r})"

#----------------------------------------

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

#----------------------------------------

engine = create_engine("sqlite:///sql.db", echo=True)

try:
  #Base.metadata.create_all(engine)
  with Session(engine) as session:
    stmt = select(User)

    for user in session.scalars(stmt):
      print(user)
except Exception as err:
  print("Tables missing Try running 'python sql/init_sqllite.py'", err) 
