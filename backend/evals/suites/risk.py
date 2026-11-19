from typing import Any

import pandas as pd

from evals.harness import run_concurrently
from evals.metrics import RISK_ORDINAL, Metric, SuiteReport, extract_numeric_claims, ordinal_auc, safe_rate
from pipeline.orchestrator import run_risk_assessment
from pipeline.preprocessing import get_dataframe
from pipeline.risk_model import get_risk_model
from pipeline.tools import get_patient_features

ORDINAL_AUC_THRESHOLD = 0.70
HIGH_RECALL_THRESHOLD = 0.50
FALSE_HIGH_RATE_THRESHOLD = 0.25
GROUNDED_RATE_THRESHOLD = 0.95
BAND_AGREEMENT_THRESHOLD = 0.95
COHORT_MISUSE_THRESHOLD = 0.10
DEFAULT_LIMIT = 30
COHORT_TOOL = "get_cohort_average"


def sample_patients(limit: int, seed: int) -> pd.DataFrame:
    df = get_dataframe()
    held_out = set(get_risk_model().test_patient_ids)
    if held_out:
        df = df[df["Patient_ID"].isin(held_out)]
    per_class = max(1, limit // 2)
    positives = df[df["Outcome"] == 1]
    negatives = df[df["Outcome"] == 0]
    positives = positives.sample(n=min(per_class, len(positives)), random_state=seed)
    negatives = negatives.sample(n=min(limit - len(positives), len(negatives)), random_state=seed)
    return pd.concat([positives, negatives])


def model_metrics() -> list[Metric]:
    model = get_risk_model()
    df = get_dataframe()
    held_out = df[df["Patient_ID"].isin(set(model.test_patient_ids))]
    scores = [model.score(get_patient_features(pid) or {}) for pid in held_out["Patient_ID"]]
    outcomes = held_out["Outcome"].tolist()
    pos = [s for s, y in zip(scores, outcomes) if y == 1]
    neg = [s for s, y in zip(scores, outcomes) if y == 0]
    return [
        Metric("model_heldout_size", float(len(scores))),
        Metric("model_ordinal_auc", ordinal_auc([RISK_ORDINAL[s.risk_band.value] for s in pos], [RISK_ORDINAL[s.risk_band.value] for s in neg]), ORDINAL_AUC_THRESHOLD),
        Metric("model_probability_auc", ordinal_auc([s.probability for s in pos], [s.probability for s in neg])),
        Metric("model_high_recall", safe_rate(sum(s.risk_band.value == "High" for s in pos), len(pos)), HIGH_RECALL_THRESHOLD),
        Metric(
            "model_false_high_rate",
            safe_rate(sum(s.risk_band.value == "High" for s in neg), len(neg)),
            FALSE_HIGH_RATE_THRESHOLD,
            higher_is_better=False,
        ),
    ]


async def _assess(row: dict[str, Any]) -> dict[str, Any]:
    patient_id = row["Patient_ID"]
    features = get_patient_features(patient_id) or {}
    case: dict[str, Any] = {"patient_id": patient_id, "outcome": int(row["Outcome"]), "features": features}
    try:
        risk_run = await run_risk_assessment(patient_id)
    except Exception as exc:
        return {**case, "error": repr(exc)}

    assessment = risk_run.assessment
    text = " ".join([*assessment.key_factors, assessment.reasoning])
    return {
        **case,
        "risk_level": assessment.risk_level.value,
        "risk_score": assessment.risk_score,
        "agent_risk_level": risk_run.agent_risk_level.value,
        "key_factors": assessment.key_factors,
        "reasoning": assessment.reasoning,
        "tools_called": sorted(risk_run.tools_called),
        "claims": [vars(c) for c in extract_numeric_claims(text, features)],
    }


async def run(limit: int | None, concurrency: int, seed: int, **_: Any) -> SuiteReport:
    sample = sample_patients(limit or DEFAULT_LIMIT, seed)
    rows = sample[["Patient_ID", "Outcome"]].to_dict("records")
    results = await run_concurrently(rows, _assess, concurrency)

    scored = [r for r in results if "error" not in r]
    positives = [r for r in scored if r["outcome"] == 1]
    negatives = [r for r in scored if r["outcome"] == 0]
    claims = [c for r in scored for c in r["claims"]]
    medium = [r for r in scored if r["risk_level"] == "Medium"]
    not_medium = [r for r in scored if r["risk_level"] != "Medium"]

    def used_cohort(r: dict[str, Any]) -> bool:
        return COHORT_TOOL in r["tools_called"]

    metrics = [
        *model_metrics(),
        Metric(
            "sample_ordinal_auc",
            ordinal_auc([RISK_ORDINAL[r["risk_level"]] for r in positives], [RISK_ORDINAL[r["risk_level"]] for r in negatives]),
        ),
        Metric("sample_high_recall", safe_rate(sum(r["risk_level"] == "High" for r in positives), len(positives))),
        Metric("sample_false_high_rate", safe_rate(sum(r["risk_level"] == "High" for r in negatives), len(negatives))),
        Metric("agent_band_agreement", safe_rate(sum(r["agent_risk_level"] == r["risk_level"] for r in scored), len(scored)), BAND_AGREEMENT_THRESHOLD),
        Metric("numeric_grounding_rate", safe_rate(sum(c["grounded"] for c in claims), len(claims)), GROUNDED_RATE_THRESHOLD),
        Metric("numeric_claims_checked", float(len(claims))),
        Metric("cohort_usage_on_medium", safe_rate(sum(used_cohort(r) for r in medium), len(medium))),
        Metric(
            "cohort_usage_outside_medium",
            safe_rate(sum(used_cohort(r) for r in not_medium), len(not_medium)),
            COHORT_MISUSE_THRESHOLD,
            higher_is_better=False,
        ),
        Metric("error_rate", safe_rate(len(results) - len(scored), len(results)), 0.0, higher_is_better=False),
    ]

    failures: list[dict[str, Any]] = []
    for r in results:
        pid = r["patient_id"]
        if "error" in r:
            failures.append({"patient_id": pid, "error": r["error"]})
            continue
        bad_claims = [c for c in r["claims"] if not c["grounded"]]
        if bad_claims:
            failures.append({"patient_id": pid, "ungrounded_claims": bad_claims})
        if r["agent_risk_level"] != r["risk_level"]:
            failures.append({"patient_id": pid, "issue": f"agent said {r['agent_risk_level']}, model band {r['risk_level']}"})
        if r["risk_level"] != "Medium" and used_cohort(r):
            failures.append({"patient_id": pid, "issue": f"cohort tool called for a {r['risk_level']} band"})

    for r in results:
        r.pop("features", None)
    return SuiteReport("risk", metrics, results, failures)
