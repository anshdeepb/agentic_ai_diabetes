from collections.abc import Iterator
from contextlib import contextmanager

from agents import SessionSettings, SQLiteSession

from config import get_settings


def get_session(session_id: str) -> SQLiteSession:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(
        session_id,
        db_path=settings.db_path,
        session_settings=SessionSettings(limit=settings.session_history_limit),
    )


@contextmanager
def open_session(session_id: str) -> Iterator[SQLiteSession]:
    session = get_session(session_id)
    try:
        yield session
    finally:
        session.close()
