import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP = Path(tempfile.mkdtemp())
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from memory.state import StateStore
from pipeline import orchestrator
from pipeline.schemas import (
    Intent,
    OrchestratorDecision,
    Recommendation,
    RiskAssessment,
    RiskLevel,
)


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.db")


@pytest.fixture
def session_id(request):
    return f"test-{request.node.name}"[:64].replace("[", "-").replace("]", "-")


class FakeRunner:
    def __init__(self):
        self.calls: list[str] = []
        self.decisions: list[OrchestratorDecision] = []

    def queue(self, intent: Intent, patient_id: str | None):
        self.decisions.append(OrchestratorDecision(intent=intent, patient_id=patient_id, reasoning="test"))

    async def run(self, agent, input, **kwargs):
        self.calls.append(agent.name)
        if agent.name == "Orchestrator":
            session = kwargs.get("session")
            if session is not None:
                await session.add_items([{"role": "user", "content": input}])
            return SimpleNamespace(final_output=self.decisions.pop(0))
        if agent.name == "Risk":
            return SimpleNamespace(
                final_output=RiskAssessment(
                    patient_id="wrong",
                    risk_level=RiskLevel.high,
                    key_factors=["Glucose 148"],
                    reasoning="High glucose.",
                    used_cohort_context=False,
                )
            )
        if agent.name == "Recommendation":
            return SimpleNamespace(
                final_output=Recommendation(
                    patient_id="wrong", priority_actions=["Cut refined carbs"], rationale="Glucose."
                )
            )
        raise AssertionError(f"unexpected agent {agent.name}")


@pytest.fixture
def fake_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(orchestrator.Runner, "run", runner.run)
    return runner
