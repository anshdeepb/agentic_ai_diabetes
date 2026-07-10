# Agentic Diabetes Risk Assistant

A multi-agent AI system that assesses diabetes risk and generates targeted health
recommendations, built with the OpenAI Agents SDK. An orchestrator agent classifies
user intent and routes requests to specialized agents, each with structured,
type-safe outputs enforced via Pydantic.

## Overview

This project explores agentic AI architecture — autonomous, tool-calling agents
that reason over data and make context-dependent decisions — applied to a
real-world healthcare use case using the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database).

Rather than a single LLM call, the system is composed of three cooperating agents:

- **Orchestrator Agent** — classifies incoming queries and routes them to the
  appropriate downstream agent, chaining agents together when needed.
- **Risk Assessment Agent** — uses tool-calling to look up patient data and,
  when a value looks borderline, autonomously decides to pull cohort-level
  context before producing a structured risk assessment.
- **Recommendation Agent** — takes a structured risk assessment and generates
  concrete, prioritized lifestyle recommendations targeted at the specific
  risk factors identified.

## Architecture

![architecture](docs/agentic_ai_diabetes.svg)>>

**Flow:**
1. User submits a natural language query (e.g. *"How risky does patient 0 look?"*)
2. Orchestrator classifies intent: risk assessment only, recommendation only,
   or recommendation requiring a fresh risk assessment first
3. Risk Agent is invoked with tool access to patient records and cohort averages,
   deciding for itself whether cohort comparison is needed
4. If a recommendation was requested, the Risk Agent's structured output is
   passed to the Recommendation Agent, which produces targeted advice
5. Final structured result is returned to the user

## Tech Stack

- **Python**
- **OpenAI Agents SDK** — agent orchestration, tool-calling, structured outputs
- **Pydantic** — schema definition and validation for all agent outputs
- **Pandas** — data cleaning and cohort/record lookups
- **GPT-4o-mini** — underlying model for all agents

## Data

Source data: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database).

Cleaning steps applied:
- Identified and handled disguised missing values (zeros used as placeholders
  for missing measurements in `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, `BMI`)
- Dropped rows with small counts of missing values (<<X>> columns, under ~5% missing)
- Median-imputed columns with larger gaps in missing data (`SkinThickness`, `Insulin`)
- Added a synthetic `patient_id` for record lookup

> Note: a Snowflake medallion architecture (bronze/silver/gold) was designed
> for this pipeline as a planned extension; the current implementation runs
> against a locally cleaned dataframe. See [Future Work](#future-work).

## Setup

1. Clone the repo:
```bash
   git clone https://github.com/anshdeepb/agentic_ai_diabetes.git
   cd agentic_ai_diabetes
```

2. Create a virtual environment and install dependencies:
```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OpenAI API key:

## Example Usage

```python
result = await orchestrate("How risky does patient 0 look?")
# → Returns a structured RiskAssessment: risk level, key factors, reasoning

result = await orchestrate("What should patient 0 do to improve their health?")
# → Chains through Risk Assessment, then returns a structured Recommendation
```

## What Makes This "Agentic"

Unlike a single prompt-response system, the Risk Assessment Agent makes an
autonomous decision mid-reasoning: after examining a patient's raw data, it
independently determines whether the values are clearly normal, clearly
high-risk, or borderline — only calling the cohort-comparison tool in the
borderline case rather than following a fixed sequence every time.

## Future Work

- Migrate the data layer to a full Snowflake medallion pipeline (bronze/silver/gold)
  for production-grade data infrastructure
- Add role-based agents (patient, doctor, scheduling staff) with access-controlled
  routing through the orchestrator
- Add a cohort comparison agent and availability/scheduling agent
- Persist conversation state so `recommendation_only` queries can reuse a prior
  risk assessment instead of always re-running it

## Author

*Anshdeep Bhachoo*