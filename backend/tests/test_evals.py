import math

import pytest

from evals.judge import JudgeVerdict
from evals.metrics import Metric, extract_numeric_claims, ordinal_auc, score_routing_turn
from evals.suites import guardrails, recommendation, risk, routing
from pipeline.schemas import Intent
from pipeline.tools import patient_exists

FEATURES = {"Glucose": 148.0, "BMI": 33.6, "BloodPressure": 72.0, "Age": 50, "Insulin": 125.5}


def test_ordinal_auc_perfect_separation():
    assert ordinal_auc([2, 2], [0, 1]) == 1.0


def test_ordinal_auc_counts_ties_as_half():
    assert ordinal_auc([1], [1]) == 0.5


def test_ordinal_auc_reversed():
    assert ordinal_auc([0], [2]) == 0.0


def test_ordinal_auc_empty_is_nan():
    assert math.isnan(ordinal_auc([], [1]))


def test_numeric_claims_grounded_when_values_match():
    claims = extract_numeric_claims("Glucose 148 mg/dL and a BMI of 33.6, age 50", FEATURES)
    assert {c.feature for c in claims} == {"Glucose", "BMI", "Age"}
    assert all(c.grounded for c in claims)


def test_numeric_claims_flag_hallucinated_values():
    claims = extract_numeric_claims("Glucose: 180 mg/dL", FEATURES)
    assert len(claims) == 1
    assert not claims[0].grounded


def test_numeric_claims_skip_cohort_reference_values():
    text = "Glucose of 148 mg/dL. Compared to the cohort average glucose of 127.97 mg/dL, this is higher."
    claims = extract_numeric_claims(text, FEATURES)
    assert [c.claimed for c in claims] == [148.0]
    assert all(c.grounded for c in claims)


def test_numeric_claims_still_catch_patient_values_after_reference_sentence():
    text = "The cohort mean is lower. The patient's glucose of 180 is high."
    claims = extract_numeric_claims(text, FEATURES)
    assert [c.claimed for c in claims] == [180.0]
    assert not claims[0].grounded


def test_numeric_claims_skip_threshold_comparisons():
    claims = extract_numeric_claims("BMI 33.6, Obese (BMI >= 30), glucose ≥ 140 threshold", FEATURES)
    assert [c.claimed for c in claims] == [33.6]


def test_numeric_claims_ignore_text_without_feature_numbers():
    assert extract_numeric_claims("Elevated glucose and high BMI", FEATURES) == []


def test_numeric_claims_handle_multiword_alias():
    claims = extract_numeric_claims("blood pressure of 72", FEATURES)
    assert claims[0].feature == "BloodPressure"
    assert claims[0].grounded


def test_routing_turn_scoring():
    result = {"type": "recommendation", "recommendation": {"patient_id": "0"}, "assessment": {"patient_id": "0"}}
    assert score_routing_turn({"type": "recommendation", "patient_id": "0"}, result) == []
    assert score_routing_turn({"type": "recommendation", "patient_id": "5"}, result)
    assert score_routing_turn({"type": "clarification"}, result)


def test_metric_threshold_direction():
    assert Metric("m", 0.9, 0.8).passed
    assert not Metric("m", 0.7, 0.8).passed
    assert Metric("err", 0.0, 0.0, higher_is_better=False).passed
    assert not Metric("err", 0.1, 0.0, higher_is_better=False).passed


def test_judge_verdict_pass_rule():
    assert JudgeVerdict(targeted=4, specific=4, prioritized=5, safe=True, issues=[]).passed
    assert not JudgeVerdict(targeted=3, specific=5, prioritized=5, safe=True, issues=[]).passed
    assert not JudgeVerdict(targeted=5, specific=5, prioritized=5, safe=False, issues=["dosage"]).passed


def test_routing_dataset_is_well_formed():
    cases = routing.load_dataset()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["turns"]
        for turn in case["turns"]:
            expect = turn["expect"]
            assert expect["type"] in {"risk_assessment", "recommendation", "clarification", "refusal"}
            if expect["type"] not in {"clarification", "refusal"}:
                assert patient_exists(expect["patient_id"]), f"{case['id']}: unknown patient"


def test_risk_sample_is_balanced():
    sample = risk.sample_patients(10, seed=1)
    assert len(sample) == 10
    assert sample["Outcome"].sum() == 5


def test_guardrail_dataset_is_well_formed():
    cases = guardrails.load_dataset()
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["expect"] for c in cases} == {"in_scope", "off_topic", "prompt_injection", "unsafe_request"}
    assert sum(c["expect"] == "in_scope" for c in cases) >= 10


async def test_guardrail_suite_runs_with_fake_runner(fake_runner, store):
    report = await guardrails.run(store=store, limit=None, concurrency=4, seed=1)
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["false_block_rate"] == 0.0
    assert metrics["error_rate"] == 0.0
    assert 0.0 < metrics["attack_block_rate"] < 1.0


@pytest.mark.parametrize("suite_name", ["routing", "risk", "recommendation"])
async def test_suites_run_end_to_end_with_fake_runner(suite_name, fake_runner, store):
    suite = {"routing": routing, "risk": risk, "recommendation": recommendation}[suite_name]
    if suite_name == "routing":
        fake_runner.queue(Intent.risk_only, "0")
    report = await suite.run(store=store, limit=1 if suite_name == "routing" else 2, concurrency=2, seed=3)
    metrics = {m.name: m.value for m in report.metrics}
    assert report.suite == suite_name
    assert report.cases
    if suite_name == "routing":
        assert metrics["turn_accuracy"] == 1.0
    elif suite_name == "risk":
        assert metrics["error_rate"] == 0.0
        assert 0.0 <= metrics["model_ordinal_auc"] <= 1.0
        assert metrics["model_heldout_size"] > 100
    else:
        assert metrics["pass_rate"] == 1.0
        assert metrics["safety_rate"] == 1.0
