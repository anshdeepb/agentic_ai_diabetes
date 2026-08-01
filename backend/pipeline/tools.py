# creating the functions that can query the dataframe
import pandas as pd
import numpy as np
from agents import Agent, function_tool
from pipeline.preprocessing import df

@function_tool
def get_patient_record(patient_id: str) -> dict:
    patient_record = df[df['Patient_ID'] == patient_id]

    if patient_record.empty:
        return {"error": "patient not found"}
    
    patient_record = patient_record.iloc[0].to_dict()

    return(patient_record)

@function_tool
def get_cohort_average(age_low: int, age_high: int) -> dict:
    cohort = df[(df['Age'] >= age_low) & (df['Age'] <= age_high)]

    if cohort.empty:
        return {"error": "no patients in this age range"}
    
    cohort_average = cohort[['Glucose', 'BMI', 'BloodPressure']].mean().to_dict()

    return cohort_average