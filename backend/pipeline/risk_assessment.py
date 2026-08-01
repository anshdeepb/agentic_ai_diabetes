from openai import OpenAI
from agents import Agent, Runner
from pipeline.tools import get_patient_record, get_cohort_average
from pipeline.schemas import RiskAssessment
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

risk_agent = Agent(
    name="Risk",
    instructions="You are a helpful patient diabetes risk analysis agent. Call get_patient_record first. Only call get_cohort_average if a value looks borderline.",
    model="gpt-4o-mini",
    tools=[get_patient_record, get_cohort_average],
    output_type=RiskAssessment
)