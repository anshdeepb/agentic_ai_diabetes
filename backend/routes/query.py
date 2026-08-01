from typing import Union
from fastapi import APIRouter
from pipeline.orchestrator import orchestrate
from pipeline.schemas import RiskAssessment, Recommendation
from schemas import QueryRequest

router = APIRouter()

@router.post("/query", response_model=Union[RiskAssessment, Recommendation])
async def query(request: QueryRequest):
    result = await orchestrate(request.query)
    return result