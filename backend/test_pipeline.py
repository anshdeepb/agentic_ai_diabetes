# backend/test_pipeline.py
import asyncio
from pipeline.orchestrator import orchestrate
from pipeline.schemas import RiskAssessment, Recommendation

def print_result(result):
    if isinstance(result, RiskAssessment):
        print(f"Patient: {result.patient_id}")
        print(f"Risk level: {result.risk_level.value}")
        print("Key factors:")
        for factor in result.key_factors:
            print(f"  - {factor}")
        print(f"Reasoning: {result.reasoning}")

    elif isinstance(result, Recommendation):
        print(f"Patient: {result.patient_id}")
        print("Priority actions:")
        for i, action in enumerate(result.priority_actions, 1):
            print(f"  {i}. {action}")
        print(f"Rationale: {result.rationale}")

    else:
        print(result)

async def main():
    result = await orchestrate("What is patient 0's risk level?")
    print_result(result)

if __name__ == "__main__":
    asyncio.run(main())