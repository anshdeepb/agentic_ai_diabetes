import json
import math
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from config import get_settings
from pipeline.schemas import RiskLevel


class Driver(BaseModel):
    feature: str
    value: float
    contribution: float


class ClinicalFlag(BaseModel):
    feature: str
    value: float
    flag: str


class RiskScore(BaseModel):
    probability: float
    risk_band: RiskLevel
    drivers: list[Driver]
    clinical_flags: list[ClinicalFlag]
    model_version: str


class RiskModel:
    def __init__(self, card: dict[str, Any]):
        self.card = card
        self.version: str = card["model_version"]
        self.features: list[str] = card["features"]
        self.mean: list[float] = card["scaler_mean"]
        self.scale: list[float] = card["scaler_scale"]
        self.coef: list[float] = card["coefficients"]
        self.intercept: float = card["intercept"]
        self.low_max: float = card["bands"]["low_max"]
        self.high_min: float = card["bands"]["high_min"]
        self.test_patient_ids: list[str] = card.get("test_patient_ids", [])

    def band(self, probability: float) -> RiskLevel:
        if probability >= self.high_min:
            return RiskLevel.high
        if probability >= self.low_max:
            return RiskLevel.medium
        return RiskLevel.low

    def score(self, features: dict[str, Any], top_n: int = 3) -> RiskScore:
        contributions = {
            name: coef * (float(features[name]) - mean) / scale
            for name, coef, mean, scale in zip(self.features, self.coef, self.mean, self.scale, strict=True)
        }
        logit = self.intercept + sum(contributions.values())
        probability = 1 / (1 + math.exp(-logit))
        drivers = [
            Driver(feature=name, value=float(features[name]), contribution=round(value, 3))
            for name, value in sorted(contributions.items(), key=lambda kv: -kv[1])
            if value > 0
        ][:top_n]
        return RiskScore(
            probability=round(probability, 4),
            risk_band=self.band(probability),
            drivers=drivers,
            clinical_flags=clinical_flags(features),
            model_version=self.version,
        )


def clinical_flags(features: dict[str, Any]) -> list[ClinicalFlag]:
    flags: list[ClinicalFlag] = []

    def add(feature: str, flag: str) -> None:
        flags.append(ClinicalFlag(feature=feature, value=float(features[feature]), flag=flag))

    glucose = float(features["Glucose"])
    if glucose >= 200:
        add("Glucose", "2-hour OGTT glucose in the diabetic range (>= 200 mg/dL)")
    elif glucose >= 140:
        add("Glucose", "2-hour OGTT glucose indicates impaired glucose tolerance (140-199 mg/dL)")

    bmi = float(features["BMI"])
    if bmi >= 30:
        add("BMI", "Obese (BMI >= 30)")
    elif bmi >= 25:
        add("BMI", "Overweight (BMI 25-29.9)")

    diastolic = float(features["BloodPressure"])
    if diastolic >= 90:
        add("BloodPressure", "Diastolic pressure in stage 2 hypertension range (>= 90 mm Hg)")
    elif diastolic >= 80:
        add("BloodPressure", "Diastolic pressure in stage 1 hypertension range (80-89 mm Hg)")

    if float(features["Age"]) >= 45:
        add("Age", "Age 45 or older, a recognised screening risk factor")

    return flags


@lru_cache
def get_risk_model() -> RiskModel:
    return RiskModel(json.loads(get_settings().risk_model_path.read_text()))
