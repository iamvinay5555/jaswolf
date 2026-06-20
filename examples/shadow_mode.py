"""Shadow-mode wrapper: run JASWOLF beside the current memory provider (Mem0)
without letting it touch the live prompt.

The primary provider answers everything; JASWOLF receives duplicate writes and
runs every recall in parallel. Each operation appends a JSONL comparison
record (schema from jasmine_feedback.md) for offline scoring. Cut over only
after the logged evidence clears the acceptance gates.

The primary is duck-typed: anything with `add(text) -> None` and
`search(query) -> list[str]`. Write a 5-line adapter for your Mem0 client.

Run the self-contained demo:  python examples/shadow_mode.py
"""

import asyncio
import json
import time
from pathlib import Path

from jaswolf import JaswolfMemoryProvider, JaswolfSettings


class ShadowMemory:
    """Primary stays authoritative; JASWOLF shadows every call and logs the diff."""

    def __init__(self, primary, shadow: JaswolfMemoryProvider, log_path: str = "shadow_log.jsonl"):
        self.primary = primary
        self.shadow = shadow
        self.log_path = Path(log_path)

    def _log(self, record: dict) -> None:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def remember(self, text: str, source: str = "explicit_remember") -> None:
        primary_status = "stored"
        try:
            await self.primary.add(text)
        except Exception:
            primary_status = "failed"

        jaswolf_status, jaswolf_type = "failed", None
        try:
            results = await self.shadow.observe([{"role": "user", "content": text}])
            if not results:
                jaswolf_status = "skipped"  # durability gate rejected it
            else:
                entry = results[0]
                jaswolf_status = "stored" if entry["created"] else "reinforced"
                jaswolf_type = entry["memory"]["memory_type"]
                if jaswolf_type == "working":
                    jaswolf_status = "working"
        except Exception:
            pass

        self._log(
            {
                "kind": "write",
                "source": source,
                "input_text": text,
                "mem0_write": primary_status,
                "jaswolf_write": jaswolf_status,
                "jaswolf_memory_type": jaswolf_type,
                "human_label": None,  # fill during review: correct|pollution|wrong_type|...
            }
        )

    async def recall(self, query: str, task_type: str = "other", top_k: int = 5) -> list[str]:
        primary_results: list[str] = []
        try:
            primary_results = await self.primary.search(query)
        except Exception:
            pass

        jaswolf_results: list[str] = []
        try:
            jaswolf_results = await self.shadow.recall(query, top_k=top_k)
        except Exception:
            pass

        self._log(
            {
                "kind": "recall",
                "query": query,
                "task_type": task_type,
                "mem0_top_results": primary_results[:top_k],
                "jaswolf_top_results": jaswolf_results,
                "winner": None,  # fill during review: mem0|jaswolf|tie|neither
                "notes": None,
            }
        )
        return primary_results  # the live prompt only ever sees the primary


# --- demo with a stub primary (replace with your Mem0 adapter) ----------------


class StubPrimary:
    def __init__(self):
        self.items: list[str] = []

    async def add(self, text: str) -> None:
        self.items.append(text)

    async def search(self, query: str) -> list[str]:
        terms = query.lower().split()
        return [t for t in self.items if any(w in t.lower() for w in terms)][:5]


async def main() -> None:
    shadow_provider = await JaswolfMemoryProvider.embedded(
        settings=JaswolfSettings(database_url="sqlite:///./shadow_demo.db", log_level="ERROR"),
        user_id="alice",
        auto_sweep=False,
    )
    memory = ShadowMemory(StubPrimary(), shadow_provider, log_path="shadow_log.jsonl")

    await memory.remember("I prefer Python for backend work.")
    await memory.remember("I like this answer, thanks honey.")   # gate should reject
    await memory.remember("I plan to go for lunch in 10 minutes.")  # gate should downroute
    results = await memory.recall("backend language preference", task_type="coding")

    print("primary returned:", results)
    print("comparison log:")
    for line in Path("shadow_log.jsonl").read_text().splitlines():
        record = json.loads(line)
        summary = {k: record[k] for k in ("kind", "mem0_write", "jaswolf_write", "jaswolf_memory_type") if k in record}
        print(" ", record.get("input_text") or record.get("query"), "->", summary)

    await shadow_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
