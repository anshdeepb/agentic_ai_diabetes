import math

import pytest

from pipeline.orchestrator import run_risk_assessment
from pipeline.risk_model import clinical_flags, get_risk_model
from pipeline.schemas import RiskLevel
from pipeline.tools import get_patient_features

BASE = {
    "Pregnancies": 1,
    "Glucose": 100.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 100.0,
    "BMI": 22.0,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 30,
}


def test_model_card_has_held_out_ids_and_metrics():
    model = get_risk_model()
    assert model.version.startswith("logreg-")
    assert len(model.test_patient_ids) > 100
    assert model.card["metrics"]["test_auc"] > 0.75
    assert model.low_max < model.high_min


def test_score_matches_logistic_formula():
    model = get_risk_model()
    features = get_patient_features("0")
    assert features is not None
    logit = model.intercept + sum(
        c * (float(features[f]) - m) / s
        for f, c, m, s in zip(model.features, model.coef, model.mean, model.scale, strict=True)
    )
    assert model.score(features).probability == pytest.approx(1 / (1 + math.exp(-logit)), abs=1e-4)


@pytest.mark.parametrize(
    ("probability", "band"),
    [
        (0.0, RiskLevel.low),
        (0.1999, RiskLevel.low),
        (0.2, RiskLevel.medium),
        (0.4499, RiskLevel.medium),
        (0.45, RiskLevel.high),
        (1.0, RiskLevel.high),
    ],
)
def test_band_boundaries(probability, band):
    assert get_risk_model().band(probability) == band


def test_higher_glucose_raises_score():
    model = get_risk_model()
    low = model.score(BASE).probability
    high = model.score({**BASE, "Glucose": 190.0}).probability
    assert high > low


def test_drivers_are_positive_contributors_only():
    score = get_risk_model().score({**BASE, "Glucose": 190.0, "BMI": 38.0})
    assert score.drivers
    assert all(d.contribution > 0 for d in score.drivers)
    assert score.drivers[0].feature == "Glucose"


def test_clinical_flags_use_ogtt_and_diastolic_thresholds():
    flags = {
        f.feature: f.flag
        for f in clinical_flags({**BASE, "Glucose": 205.0, "BMI": 31.0, "BloodPressure": 92.0, "Age": 50})
    }
    assert "diabetic range" in flags["Glucose"]
    assert "Obese" in flags["BMI"]
    assert "stage 2" in flags["BloodPressure"]
    assert "Age" in flags
    assert clinical_flags(BASE) == []


def test_impaired_glucose_tolerance_band():
    flags = clinical_flags({**BASE, "Glucose": 150.0})
    assert "impaired glucose tolerance" in flags[0].flag


async def test_final_level_follows_model_even_if_agent_disagrees(fake_runner):
    fake_runner.risk_agent_level = RiskLevel.low
    run = await run_risk_assessment("0")
    features = get_patient_features("0")
    assert features is not None
    expected = get_risk_model().score(features)
    assert run.assessment.risk_level == expected.risk_band
    assert run.assessment.risk_score == expected.probability
    assert run.agent_risk_level == RiskLevel.low


async def test_cohort_flag_comes_from_actual_tool_calls(fake_runner):
    run = await run_risk_assessment("0")
    assert run.assessment.used_cohort_context is False

    fake_runner.risk_tools_called = ["get_cohort_average"]
    run = await run_risk_assessment("0")
    assert run.assessment.used_cohort_context is True
    assert run.tools_called == {"get_cohort_average"}


async def test_unknown_patient_raises(fake_runner):
    with pytest.raises(ValueError, match="not found"):
        await run_risk_assessment("99999")
