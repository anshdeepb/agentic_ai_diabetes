import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from memory.state import StateStore
from pipeline.orchestrator import orchestrate

QUERIES = [
    "What is patient 0's risk level?",
    "What should they do about it?",
    "How about patient 5?",
]


async def main() -> None:
    store = StateStore(get_settings().db_path)
    session_id = f"smoke-{uuid.uuid4().hex[:12]}"
    for query in QUERIES:
        result = await orchestrate(query, session_id, store)
        print(f"> {query}\n{result.model_dump_json(indent=2)}\n")


if __name__ == "__main__":
    asyncio.run(main())
