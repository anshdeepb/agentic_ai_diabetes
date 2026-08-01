import pandas as pd
import numpy as np

def load_data() -> pd.DataFrame:
    df = pd.read_csv('./data/archive/diabetes.csv')
    df = df.dropna().drop_duplicates()

    zeroes_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    df[zeroes_as_missing_cols] = df[zeroes_as_missing_cols].replace(0, np.nan)

    df = df.dropna(subset='Glucose')
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
    df = df.dropna(subset='BMI')

    df = df.reset_index(drop=True)
    df['Patient_ID'] = df.index.astype(str)

    return df

df = load_data()
