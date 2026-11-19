from pydantic import BaseModel, Field

from config import get_settings
from pipeline.schemas import PipelineResult


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=get_settings().max_query_chars)


class QueryResponse(BaseModel):
    session_id: str
    result: PipelineResult
