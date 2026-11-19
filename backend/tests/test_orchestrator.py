from memory.session import get_session
from pipeline.orchestrator import orchestrate
from pipeline.schemas import (
    ClarificationResult,
    Intent,
    RecommendationResult,
    RiskAssessmentResult,
)


async def test_risk_only_runs_risk_agent(fake_runner, store, session_id):
    fake_runner.queue(Intent.risk_only, "0")
    result = await orchestrate("risk for patient 0?", session_id, store)
    assert isinstance(result, RiskAssessmentResult)
    assert result.assessment.patient_id == "0"
    assert not result.from_cache
    assert fake_runner.calls == ["InputGuard", "Orchestrator", "Risk"]


async def test_follow_up_recommendation_reuses_cached_assessment(fake_runner, store, session_id):
    fake_runner.queue(Intent.risk_only, "0")
    await orchestrate("risk for patient 0?", session_id, store)

    fake_runner.queue(Intent.recommendation, None)
    result = await orchestrate("what should they do?", session_id, store)

    assert isinstance(result, RecommendationResult)
    assert result.from_cache
    assert result.recommendation.patient_id == "0"
    assert fake_runner.calls == ["InputGuard", "Orchestrator", "Risk", "InputGuard", "Orchestrator", "Recommendation"]


async def test_missing_patient_asks_for_clarification(fake_runner, store, session_id):
    fake_runner.queue(Intent.risk_only, None)
    result = await orchestrate("how risky is it?", session_id, store)
    assert isinstance(result, ClarificationResult)
    assert fake_runner.calls == ["InputGuard", "Orchestrator"]


async def test_unknown_patient_asks_for_clarification(fake_runner, store, session_id):
    fake_runner.queue(Intent.risk_only, "99999")
    result = await orchestrate("risk for patient 99999", session_id, store)
    assert isinstance(result, ClarificationResult)
    assert "99999" in result.message
    assert (await store.load(session_id)).active_patient_id is None


async def test_conversation_history_stores_compact_summary(fake_runner, store, session_id):
    session = get_session(session_id)
    await session.clear_session()
    fake_runner.queue(Intent.risk_only, "0")
    await orchestrate("risk for patient 0?", session_id, store)
    items = await session.get_items()
    assert items[-1]["role"] == "assistant"
    assert items[-1]["content"].startswith("Patient 0: High risk")
