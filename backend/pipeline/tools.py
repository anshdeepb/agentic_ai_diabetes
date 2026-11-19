import hashlib
import json
from typing import Any

from agents import function_tool

from pipeline.preprocessing import FEATURE_COLUMNS, get_dataframe


def _to_native(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def patient_exists(patient_id: str) -> bool:
    df = get_dataframe()
    return bool((df["Patient_ID"] == patient_id).any())


def get_patient_features(patient_id: str) -> dict[str, Any] | None:
    df = get_dataframe()
    rows = df[df["Patient_ID"] == patient_id]
    if rows.empty:
        return None
    return {col: _to_native(val) for col, val in rows.iloc[0][FEATURE_COLUMNS].items()}


def record_hash(patient_id: str) -> str:
    features = get_patient_features(patient_id)
    payload = json.dumps({"patient_id": patient_id, "features": features}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cohort_average(age_low: int, age_high: int) -> dict[str, Any]:
    df = get_dataframe()
    cohort = df[(df["Age"] >= age_low) & (df["Age"] <= age_high)]
    if cohort.empty:
        return {"error": "no patients in this age range"}
    averages = cohort[["Glucose", "BMI", "BloodPressure"]].mean()
    return {
        "cohort_size": len(cohort),
        **{col: round(_to_native(val), 2) for col, val in averages.items()},
    }


@function_tool
def get_patient_record(patient_id: str) -> dict[str, Any]:
    """Return the clinical measurements for one patient."""
    features = get_patient_features(patient_id)
    if features is None:
        return {"error": f"patient {patient_id} not found"}
    return {"patient_id": patient_id, **features}


@function_tool
def get_cohort_average(age_low: int, age_high: int) -> dict[str, Any]:
    """Return average Glucose, BMI and BloodPressure for patients aged age_low to age_high inclusive."""
    return cohort_average(age_low, age_high)
