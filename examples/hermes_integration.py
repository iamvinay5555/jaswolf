"""Wiring JASWOLF into a Hermes-style agent loop.

The pattern is three calls:

  1. `inject_context(messages)` before the LLM call — memory block lands in
     the system prompt.
  2. your LLM call (Hermes inference, OpenAI-compatible, whatever).
  3. `observe(new_turns)` after the exchange — durable facts are extracted
     and stored; repeats reinforce instead of duplicating.

Run:  python examples/hermes_integration.py
"""

import asyncio
from typing import Any

from jaswolf import JaswolfMemoryProvider, JaswolfSettings


class HermesAgent:
    """Minimal stand-in for a Hermes agent. Replace `_infer` with your real
    inference call; the memory wiring stays identical."""

    def __init__(self, memory: JaswolfMemoryProvider):
        self.memory = memory
        self.system_prompt = "You are Hermes, an autonomous agent."

    async def _infer(self, messages: list[dict[str, Any]]) -> str:
        # --- replace with your actual LLM call -------------------------------
        # resp = await openai_client.chat.completions.create(
        #     model="Hermes-3-Llama-3.1-70B", messages=messages
        # )
        # return resp.choices[0].message.content
        has_memory = "What I remember" in messages[0]["content"]
        return (
            f"(model saw {'memory context ✓' if has_memory else 'no memory'}; "
            f"{len(messages)} messages)"
        )

    async def chat(self, session_id: str, history: list[dict[str, str]], user_input: str) -> str:
        history.append({"role": "user", "content": user_input})

        # 1) build the prompt with memory injected
        messages = [{"role": "system", "content": self.system_prompt}, *history]
        messages = await self.memory.inject_context(messages, token_budget=1200)

        # 2) call the model
        reply = await self._infer(messages)
        history.append({"role": "assistant", "content": reply})

        # 3) learn from the exchange (fire-and-forget is fine in production)
        await self.memory.observe(history[-2:], session_id=session_id)
        return reply


async def main() -> None:
    # Embedded mode: the engine lives inside the agent process.
    memory = await JaswolfMemoryProvider.embedded(
        settings=JaswolfSettings(database_url="sqlite:///./hermes_memory.db", log_level="WARNING"),
        user_id="alice",
        agent_id="hermes-main",
    )
    # Remote mode against a shared JASWOLF service is one line instead (synchronous, no await):
    # memory = JaswolfMemoryProvider.remote(base_url="http://localhost:8400", api_key="...", user_id="alice")

    agent = HermesAgent(memory)
    history: list[dict[str, str]] = []

    print(await agent.chat("session-1", history, "I prefer Python for backend work."))
    print(await agent.chat("session-1", history, "I'm planning to launch a SaaS by December."))

    # ...days later, a fresh session: the agent still knows
    fresh_history: list[dict[str, str]] = []
    print(await agent.chat("session-2", fresh_history, "What should I build this week?"))

    print("\nwhat the memory engine knows now:")
    for line in await memory.recall("user plans and preferences", top_k=5):
        print(f"  - {line}")

    # periodic maintenance (run from cron or the built-in sweeper)
    report = await memory.consolidate_memories(dry_run=True)
    print(f"\nconsolidation dry-run: {report['clusters_found']} mergeable clusters")

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
