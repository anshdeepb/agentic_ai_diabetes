import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

from agents import Runner

from evals.metrics import SuiteReport
from pipeline.schemas import Recommendation, RiskAssessment, RiskExplanation, RiskLevel

T = TypeVar("T")
R = TypeVar("R")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


async def run_concurrently(
    items: Iterable[T], worker: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return await asyncio.gather(*(guarded(item) for item in items))


@contextmanager
def stub_downstream_agents() -> Iterator[None]:
    real_run = Runner.run

    async def run(agent, input, **kwargs):
        if agent.name == "Risk":
            band = json.loads(str(input))["model_score"]["risk_band"]
            return _Result(RiskExplanation(risk_level=RiskLevel(band), key_factors=["stub"], reasoning="stub"))
        if agent.name == "Recommendation":
            assessment = RiskAssessment.model_validate_json(str(input))
            return _Result(
                Recommendation(patient_id=assessment.patient_id, priority_actions=["stub"], rationale="stub")
            )
        return await real_run(agent, input, **kwargs)

    with patch.object(Runner, "run", run):
        yield


class _Result:
    def __init__(self, final_output: Any):
        self.final_output = final_output
        self.new_items: list[Any] = []


def print_report(report: SuiteReport, max_failures: int = 10) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"\n=== {report.suite} [{status}] ({len(report.cases)} cases)")
    width = max(len(m.name) for m in report.metrics)
    for m in report.metrics:
        if m.threshold is None:
            bar = ""
        else:
            op = ">=" if m.higher_is_better else "<="
            bar = f"  (threshold {op} {m.threshold:.2f}) {'ok' if m.passed else 'BELOW'}"
        print(f"  {m.name.ljust(width)}  {m.value:.3f}{bar}")
    if report.failures:
        print(f"  failures ({len(report.failures)}):")
        for failure in report.failures[:max_failures]:
            print(f"    - {json.dumps(failure, default=str)[:300]}")
        if len(report.failures) > max_failures:
            print(f"    ... {len(report.failures) - max_failures} more in the JSON report")


def write_report(reports: list[SuiteReport], run_config: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{stamp}.json"
    payload = {
        "created_at": stamp,
        "config": run_config,
        "passed": all(r.passed for r in reports),
        "suites": [
            {
                "suite": r.suite,
                "passed": r.passed,
                "metrics": [{**asdict(m), "passed": m.passed} for m in r.metrics],
                "failures": r.failures,
                "cases": r.cases,
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
