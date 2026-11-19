from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_settings

ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Patient_ID"] = df.index.astype(str)

    df = df.dropna(subset=FEATURE_COLUMNS).drop_duplicates(subset=FEATURE_COLUMNS)
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)

    df = df.dropna(subset=["Glucose", "BMI"])
    for col in ["BloodPressure", "SkinThickness", "Insulin"]:
        df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


@lru_cache
def get_dataframe() -> pd.DataFrame:
    return load_data(get_settings().data_path)
