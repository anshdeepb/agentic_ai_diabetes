from typing import Literal

from config import get_settings

Role = Literal["orchestrator", "risk", "recommendation", "judge"]


def get_model(role: Role) -> str:
    settings = get_settings()
    if role == "judge":
        return settings.judge_model
    override = {
        "orchestrator": settings.orchestrator_model,
        "risk": settings.risk_model,
        "recommendation": settings.recommendation_model,
    }[role]
    return override or settings.default_model
