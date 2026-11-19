import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

_EVAL_DB = Path(tempfile.mkdtemp(prefix="evals-")) / "evals.db"
os.environ["DB_PATH"] = str(_EVAL_DB)

from config import get_settings
from evals.harness import print_report, write_report
from evals.suites import guardrails, recommendation, risk, routing
from memory.state import StateStore
from pipeline.models import get_model

SUITES = {"routing": routing, "guardrails": guardrails, "risk": risk, "recommendation": recommendation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description="Run the agent evaluation suites.")
    parser.add_argument("suites", nargs="*", choices=[*SUITES, "all"], default=["all"])
    parser.add_argument("--limit", type=int, default=None, help="Max cases per suite")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed for the risk and recommendation suites")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    selected = list(SUITES) if "all" in args.suites else args.suites
    store = StateStore(get_settings().db_path)

    run_config = {
        "suites": selected,
        "limit": args.limit,
        "seed": args.seed,
        "models": {role: get_model(role) for role in ("orchestrator", "risk", "recommendation", "guard", "judge")},
    }
    print(f"Running {', '.join(selected)} with models {run_config['models']}")

    reports = []
    for name in selected:
        report = await SUITES[name].run(store=store, limit=args.limit, concurrency=args.concurrency, seed=args.seed)
        print_report(report)
        reports.append(report)

    path = write_report(reports, run_config)
    passed = all(r.passed for r in reports)
    print(f"\n{'ALL SUITES PASSED' if passed else 'SOME SUITES FAILED'}. Report: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
