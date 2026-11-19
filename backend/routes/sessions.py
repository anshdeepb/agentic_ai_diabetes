from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, status

from memory.session import open_session
from memory.state import StateStore
from pipeline.orchestrator import orchestrate
from schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{8,64}$")]


def get_store(request: Request) -> StateStore:
    store: StateStore = request.app.state.store
    return store


@router.post("/{session_id}/messages", response_model=QueryResponse)
async def post_message(
    session_id: SessionId,
    body: QueryRequest,
    store: Annotated[StateStore, Depends(get_store)],
) -> QueryResponse:
    result = await orchestrate(body.query.strip(), session_id, store)
    return QueryResponse(session_id=session_id, result=result)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: SessionId,
    store: Annotated[StateStore, Depends(get_store)],
) -> None:
    with open_session(session_id) as session:
        await session.clear_session()
    await store.delete(session_id)
