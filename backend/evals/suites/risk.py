from typing import Any

import pandas as pd

from evals.harness import run_concurrently
from evals.metrics import RISK_ORDINAL, Metric, SuiteReport, extract_numeric_claims, ordinal_auc, safe_rate
from memory.state import ConversationState
from pipeline.orchestrator import get_or_create_assessment
from pipeline.preprocessing import get_dataframe
from pipeline.tools import get_patient_features

ORDINAL_AUC_THRESHOLD = 0.70
HIGH_RECALL_THRESHOLD = 0.50
FALSE_HIGH_RATE_THRESHOLD = 0.35
GROUNDED_RATE_THRESHOLD = 0.95
DEFAULT_LIMIT = 30


def sample_patients(limit: int, seed: int) -> pd.DataFrame:
    df = get_dataframe()
    per_class = max(1, limit // 2)
    positives = df[df["Outcome"] == 1].sample(n=min(per_class, (df["Outcome"] == 1).sum()), random_state=seed)
    negatives = df[df["Outcome"] == 0].sample(n=min(limit - len(positives), (df["Outcome"] == 0).sum()), random_state=seed)
    return pd.concat([positives, negatives])


async def _assess(row: dict[str, Any]) -> dict[str, Any]:
    patient_id = row["Patient_ID"]
    features = get_patient_features(patient_id) or {}
    case: dict[str, Any] = {"patient_id": patient_id, "outcome": int(row["Outcome"]), "features": features}
    try:
        state = ConversationState(session_id=f"eval-risk-{patient_id}")
        assessment, _ = await get_or_create_assessment(patient_id, state)
    except Exception as exc:
        return {**case, "error": repr(exc)}

    text = " ".join([*assessment.key_factors, assessment.reasoning])
    claims = extract_numeric_claims(text, features)
    return {
        **case,
        "risk_level": assessment.risk_level.value,
        "key_factors": assessment.key_factors,
        "reasoning": assessment.reasoning,
        "used_cohort_context": assessment.used_cohort_context,
        "claims": [vars(c) for c in claims],
    }


async def run(limit: int | None, concurrency: int, seed: int, **_: Any) -> SuiteReport:
    sample = sample_patients(limit or DEFAULT_LIMIT, seed)
    rows = sample[["Patient_ID", "Outcome"]].to_dict("records")
    results = await run_concurrently(rows, _assess, concurrency)

    scored = [r for r in results if "error" not in r]
    positives = [RISK_ORDINAL[r["risk_level"]] for r in scored if r["outcome"] == 1]
    negatives = [RISK_ORDINAL[r["risk_level"]] for r in scored if r["outcome"] == 0]
    claims = [c for r in scored for c in r["claims"]]
    grounded = sum(1 for c in claims if c["grounded"])

    metrics = [
        Metric("ordinal_auc", ordinal_auc(positives, negatives), ORDINAL_AUC_THRESHOLD),
        Metric("high_recall_on_diabetic", safe_rate(sum(1 for s in positives if s == 2), len(positives)), HIGH_RECALL_THRESHOLD),
        Metric(
            "false_high_rate_on_non_diabetic",
            safe_rate(sum(1 for s in negatives if s == 2), len(negatives)),
            FALSE_HIGH_RATE_THRESHOLD,
            higher_is_better=False,
        ),
        Metric("numeric_grounding_rate", safe_rate(grounded, len(claims)), GROUNDED_RATE_THRESHOLD),
        Metric("numeric_claims_checked", float(len(claims))),
        Metric("cohort_tool_usage_rate", safe_rate(sum(1 for r in scored if r["used_cohort_context"]), len(scored))),
        Metric("error_rate", safe_rate(len(results) - len(scored), len(results)), 0.0, higher_is_better=False),
    ]

    failures: list[dict[str, Any]] = []
    for r in results:
        if "error" in r:
            failures.append({"patient_id": r["patient_id"], "error": r["error"]})
            continue
        bad_claims = [c for c in r["claims"] if not c["grounded"]]
        if bad_claims:
            failures.append({"patient_id": r["patient_id"], "ungrounded_claims": bad_claims})
        if r["outcome"] == 1 and r["risk_level"] == "Low":
            failures.append({"patient_id": r["patient_id"], "issue": "diabetic patient rated Low"})
        if r["outcome"] == 0 and r["risk_level"] == "High":
            failures.append({"patient_id": r["patient_id"], "issue": "non-diabetic patient rated High"})

    for r in results:
        r.pop("features", None)
    return SuiteReport("risk", metrics, results, failures)
