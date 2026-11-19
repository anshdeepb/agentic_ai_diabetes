from pipeline.preprocessing import get_dataframe
from pipeline.tools import cohort_average, get_patient_features, patient_exists, record_hash


def test_patient_features_exclude_outcome_label():
    features = get_patient_features("0")
    assert features is not None
    assert "Outcome" not in features
    assert "Patient_ID" not in features
    assert features["Glucose"] == 148.0


def test_patient_ids_follow_source_rows():
    df = get_dataframe()
    assert "Outcome" in df.columns
    assert df["Patient_ID"].is_unique
    assert patient_exists("0")
    assert not patient_exists("99999")


def test_missing_patient_returns_none():
    assert get_patient_features("99999") is None


def test_record_hash_is_stable_and_distinct():
    assert record_hash("0") == record_hash("0")
    assert record_hash("0") != record_hash("1")


def test_cohort_average_values_are_json_native():
    result = cohort_average(45, 55)
    assert result["cohort_size"] > 0
    assert all(type(v) in (int, float) for v in result.values())


def test_cohort_average_empty_range():
    assert "error" in cohort_average(200, 210)
