import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import (
    GuardrailFunctionOutput,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
    ToolCallItem,
)

_TMP = Path(tempfile.mkdtemp())
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from memory.state import StateStore
from pipeline.guardrails import clinical_safety_guardrail
from pipeline import orchestrator
from pipeline.schemas import (
    Intent,
    OrchestratorDecision,
    Recommendation,
    RiskExplanation,
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
        self.risk_agent_level: RiskLevel | None = None
        self.risk_tools_called: list[str] = []
        self.guard_category: str = "in_scope"
        self.output_trips: dict[str, int] = {}

    def queue(self, intent: Intent, patient_id: str | None):
        self.decisions.append(OrchestratorDecision(intent=intent, patient_id=patient_id, reasoning="test"))

    async def run(self, agent, input, **kwargs):
        self.calls.append(agent.name)
        if isinstance(input, list):
            input = input[0]["content"]
        if self.output_trips.get(agent.name, 0) > 0:
            self.output_trips[agent.name] -= 1
            raise OutputGuardrailTripwireTriggered(
                OutputGuardrailResult(
                    guardrail=clinical_safety_guardrail,
                    agent_output=None,
                    agent=agent,
                    output=GuardrailFunctionOutput(output_info={"violations": ["drug_name"]}, tripwire_triggered=True),
                )
            )
        if agent.name == "InputGuard":
            from pipeline.guardrails import InputVerdict

            return SimpleNamespace(final_output=InputVerdict(category=self.guard_category, reason="test"))
        if agent.name == "Orchestrator":
            session = kwargs.get("session")
            if session is not None:
                await session.add_items([{"role": "user", "content": input}])
            return SimpleNamespace(final_output=self.decisions.pop(0))
        if agent.name == "Risk":
            band = json.loads(input)["model_score"]["risk_band"]
            return SimpleNamespace(
                final_output=RiskExplanation(
                    risk_level=self.risk_agent_level or RiskLevel(band),
                    key_factors=["Glucose 148"],
                    reasoning="High glucose.",
                ),
                new_items=[
                    ToolCallItem(agent=agent, raw_item={"type": "function_call", "name": n, "call_id": n, "arguments": "{}"})
                    for n in self.risk_tools_called
                ],
            )
        if agent.name == "Recommendation":
            return SimpleNamespace(
                final_output=Recommendation(
                    patient_id="wrong", priority_actions=["Cut refined carbs", "Walk 30 minutes daily"], rationale="Glucose."
                )
            )
        if agent.name == "RecommendationJudge":
            from evals.judge import JudgeVerdict

            return SimpleNamespace(
                final_output=JudgeVerdict(targeted=5, specific=4, prioritized=4, safe=True, issues=[])
            )
        raise AssertionError(f"unexpected agent {agent.name}")


@pytest.fixture
def fake_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(orchestrator.Runner, "run", runner.run)
    return runner
