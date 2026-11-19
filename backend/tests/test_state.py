from memory.state import CachedAssessment, ConversationState
from pipeline.schemas import RiskAssessment, RiskLevel


def _assessment() -> RiskAssessment:
    return RiskAssessment(
        patient_id="3", risk_level=RiskLevel.low, key_factors=[], reasoning="ok", used_cohort_context=False
    )


async def test_new_session_has_empty_state(store):
    state = await store.load("unknown-session")
    assert state.active_patient_id is None
    assert state.assessments == {}


async def test_state_round_trip(store):
    state = ConversationState(session_id="abc12345", active_patient_id="3")
    state.assessments["3"] = CachedAssessment(assessment=_assessment(), record_hash="h", prompt_version="v", model="m")
    await store.save(state)
    loaded = await store.load("abc12345")
    assert loaded.active_patient_id == "3"
    assert loaded.assessments["3"].assessment.risk_level == RiskLevel.low


async def test_delete_clears_state(store):
    await store.save(ConversationState(session_id="abc12345", active_patient_id="3"))
    await store.delete("abc12345")
    assert (await store.load("abc12345")).active_patient_id is None


def test_cache_invalidates_on_any_key_change():
    cached = CachedAssessment(assessment=_assessment(), record_hash="h", prompt_version="v", model="m")
    assert cached.matches("h", "v", "m")
    assert not cached.matches("other", "v", "m")
    assert not cached.matches("h", "v2", "m")
    assert not cached.matches("h", "v", "gpt-other")


def test_summary_mentions_active_and_cached_patients():
    state = ConversationState(session_id="abc12345", active_patient_id="7")
    assert "Active patient: 7" in state.summary()
    assert "assessment: none" in state.summary()


def test_decision_normalizes_patient_id():
    from pipeline.schemas import Intent, OrchestratorDecision

    def make(pid):
        return OrchestratorDecision(intent=Intent.risk_only, patient_id=pid, reasoning="")

    assert make("null").patient_id is None
    assert make("").patient_id is None
    assert make(None).patient_id is None
    assert make("patient 5").patient_id == "5"
    assert make(" 42 ").patient_id == "42"
