import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.schemas import RiskAssessment


class CachedAssessment(BaseModel):
    assessment: RiskAssessment
    record_hash: str
    prompt_version: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def matches(self, record_hash: str, prompt_version: str, model: str) -> bool:
        return self.record_hash == record_hash and self.prompt_version == prompt_version and self.model == model


class ConversationState(BaseModel):
    session_id: str
    active_patient_id: str | None = None
    assessments: dict[str, CachedAssessment] = Field(default_factory=dict)

    def summary(self) -> str:
        active = self.active_patient_id or "none"
        cached = ", ".join(sorted(self.assessments)) or "none"
        return f"Active patient: {active}. Patients with an existing risk assessment: {cached}."


class StateStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conversation_state ("
                " session_id TEXT PRIMARY KEY,"
                " state_json TEXT NOT NULL,"
                " updated_at TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _load(self, session_id: str) -> ConversationState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM conversation_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return ConversationState(session_id=session_id)
        return ConversationState.model_validate_json(row[0])

    def _save(self, state: ConversationState) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_state (session_id, state_json, updated_at)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                " state_json = excluded.state_json, updated_at = excluded.updated_at",
                (state.session_id, state.model_dump_json(), datetime.now(UTC).isoformat()),
            )

    def _delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_state WHERE session_id = ?", (session_id,))

    async def load(self, session_id: str) -> ConversationState:
        return await asyncio.to_thread(self._load, session_id)

    async def save(self, state: ConversationState) -> None:
        await asyncio.to_thread(self._save, state)

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete, session_id)
