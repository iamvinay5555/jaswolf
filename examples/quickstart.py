"""JASWOLF quickstart: embedded provider, no server, no external services.

    python examples/quickstart.py
"""

import asyncio

from jaswolf import JaswolfMemoryProvider, JaswolfSettings


async def main() -> None:
    provider = await JaswolfMemoryProvider.embedded(
        settings=JaswolfSettings(database_url="sqlite:///./quickstart.db", log_level="WARNING"),
        user_id="alice",
        auto_sweep=False,
    )

    # 1. The agent observes a conversation -> memories are extracted automatically
    observed = await provider.observe(
        [
            {"role": "user", "content": "I love Python and I prefer dark mode in every editor."},
            {"role": "user", "content": "Sarah is my cofounder. We plan to launch our SaaS by December."},
            {"role": "user", "content": "My company uses Kubernetes on Hetzner."},
        ]
    )
    print(f"extracted {len(observed)} memories:")
    for entry in observed:
        m = entry["memory"]
        print(f"  [{m['memory_type']:<12}] {m['content']}  (importance {m['importance']:.2f})")

    # 2. Explicit memory
    await provider.remember("User's Hermes agent runs on a 4GB VPS")

    # 3. Recall
    print("\nrecall('what does the user like?'):")
    for line in await provider.recall("what does the user like?"):
        print(f"  - {line}")

    # 4. Build LLM-ready context
    block = await provider.build_context(query="help me plan the product launch", token_budget=400)
    print("\ncontext block for the system prompt:\n")
    print(block)

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
