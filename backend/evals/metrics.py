import re
from dataclasses import dataclass, field
from typing import Any

RISK_ORDINAL = {"Low": 0, "Medium": 1, "High": 2}

FEATURE_ALIASES: dict[str, list[str]] = {
    "Glucose": ["glucose"],
    "BMI": ["bmi", "body mass index"],
    "BloodPressure": ["blood pressure", "bloodpressure", "bp"],
    "Insulin": ["insulin"],
    "Age": ["age"],
    "Pregnancies": ["pregnancies"],
    "SkinThickness": ["skin thickness", "skinthickness"],
    "DiabetesPedigreeFunction": ["diabetes pedigree function", "diabetespedigreefunction", "pedigree"],
}

_NUMBER_AFTER = r"(\W{0,4}(?:of|is|at|was|=|:)?\s*)(\d+(?:\.\d+)?)"
_COMPARISON = re.compile(r"[<>≤≥]")
_REFERENCE_CONTEXT = re.compile(r"\b(?:average|averages|mean|cohort|typical|median)\b")
_REFERENCE_WINDOW = 30


@dataclass
class NumericClaim:
    feature: str
    claimed: float
    actual: float
    grounded: bool


@dataclass
class Metric:
    name: str
    value: float
    threshold: float | None = None
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        if self.threshold is None:
            return True
        return self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold


@dataclass
class SuiteReport:
    suite: str
    metrics: list[Metric]
    cases: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)


def ordinal_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _values_match(claimed: float, actual: float) -> bool:
    return abs(claimed - actual) <= max(0.51, abs(actual) * 0.01)


def extract_numeric_claims(text: str, features: dict[str, Any]) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    lowered = text.lower()
    for feature, aliases in FEATURE_ALIASES.items():
        if feature not in features:
            continue
        actual = float(features[feature])
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = rf"\b{re.escape(alias)}\b{_NUMBER_AFTER}"
            matches = list(re.finditer(pattern, lowered))
            for match in matches:
                window = lowered[max(0, match.start() - _REFERENCE_WINDOW) : match.start()]
                preceding = re.split(r"[.;!?]", window)[-1]
                if _REFERENCE_CONTEXT.search(preceding):
                    continue
                if _COMPARISON.search(match.group(1)):
                    continue
                claimed = float(match.group(2))
                claims.append(NumericClaim(feature, claimed, actual, _values_match(claimed, actual)))
            if matches:
                break
    return claims


def score_routing_turn(expected: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("type") != expected["type"]:
        errors.append(f"type: expected {expected['type']}, got {result.get('type')}")
    expected_pid = expected.get("patient_id")
    if expected_pid is not None:
        actual_pid = _result_patient_id(result)
        if actual_pid != expected_pid:
            errors.append(f"patient_id: expected {expected_pid}, got {actual_pid}")
    return errors


def _result_patient_id(result: dict[str, Any]) -> str | None:
    if result.get("type") == "recommendation":
        return str(result["recommendation"]["patient_id"])
    if result.get("type") == "risk_assessment":
        return str(result["assessment"]["patient_id"])
    return None
