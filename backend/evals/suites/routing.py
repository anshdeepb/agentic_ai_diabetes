import json
import uuid
from pathlib import Path
from typing import Any

from evals.harness import run_concurrently, stub_downstream_agents
from evals.metrics import Metric, SuiteReport, safe_rate, score_routing_turn
from memory.state import StateStore
from pipeline.orchestrator import orchestrate

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "routing.json"

TURN_ACCURACY_THRESHOLD = 0.90
CONVERSATION_PASS_THRESHOLD = 0.85


def load_dataset() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(DATASET.read_text())
    return cases


async def _run_conversation(case: dict[str, Any], store: StateStore) -> dict[str, Any]:
    session_id = f"eval-{uuid.uuid4().hex[:16]}"
    turns = []
    for turn in case["turns"]:
        try:
            result = (await orchestrate(turn["user"], session_id, store)).model_dump(mode="json")
            errors = score_routing_turn(turn["expect"], result)
        except Exception as exc:
            result, errors = {"error": repr(exc)}, [f"exception: {exc!r}"]
        turns.append({"user": turn["user"], "expect": turn["expect"], "result": result, "errors": errors})
    return {"id": case["id"], "tags": case["tags"], "turns": turns}


async def run(store: StateStore, limit: int | None, concurrency: int, **_: Any) -> SuiteReport:
    cases = load_dataset()[:limit] if limit else load_dataset()

    with stub_downstream_agents():
        results = await run_concurrently(cases, lambda c: _run_conversation(c, store), concurrency)

    all_turns = [t for r in results for t in r["turns"]]
    turns_ok = sum(1 for t in all_turns if not t["errors"])
    conversations_ok = sum(1 for r in results if all(not t["errors"] for t in r["turns"]))

    metrics = [
        Metric("turn_accuracy", safe_rate(turns_ok, len(all_turns)), TURN_ACCURACY_THRESHOLD),
        Metric("conversation_pass_rate", safe_rate(conversations_ok, len(results)), CONVERSATION_PASS_THRESHOLD),
    ]
    for tag in sorted({tag for r in results for tag in r["tags"]}):
        tagged = [t for r in results if tag in r["tags"] for t in r["turns"]]
        metrics.append(
            Metric(f"turn_accuracy[{tag}]", safe_rate(sum(1 for t in tagged if not t["errors"]), len(tagged)))
        )

    failures = [
        {"id": r["id"], "user": t["user"], "errors": t["errors"]} for r in results for t in r["turns"] if t["errors"]
    ]
    return SuiteReport("routing", metrics, results, failures)
