# JasWolf Memory 🐺

<p align="center">
  <img src="assets/logo.png" width="400" alt="JasWolf Logo"/>
</p>

**Self-hosted, embeddable long-term memory for AI agents. Built for companionship — so your AI never forgets who you are.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

```python
pip install jaswolf
```

```python
from jaswolf import JaswolfMemoryProvider

memory = await JaswolfMemoryProvider.embedded(
    database_url="sqlite:///./memory.db",
    user_id="your-name",
    agent_id="your-agent",
)

# Every turn, inject what's relevant
await memory.inject_context(messages, token_budget=1200)

# Every turn, observe what happened
await memory.observe(history[-2:])
```

---

## Why JasWolf?

Most AI agents have **no persistent memory**. Every conversation starts from scratch. The agent doesn't remember your name, your preferences, what you worked on yesterday, or the fact that you prefer green tea.

JasWolf solves this — it's a **lightweight, self-hosted memory engine** that runs alongside your AI agent and remembers everything important across sessions.

**What makes JasWolf different:**

| Feature | JasWolf | Mem0 | Other |
|---|---|---|---|
| Self-hosted (no cloud dependency) | ✅ | ❌ | Varies |
| Deterministic extraction rules | ✅ | LLM-only | Varies |
| Multi-agent shared memory | ✅ | ❌ | ✅ Letta |
| Write-ahead journal (crash-proof) | ✅ | ❌ | ❌ |
| Corpus-calibrated context gate | ✅ | ❌ | ❌ |
| Memory supersession (auto-merge) | ✅ | ✅ | Varies |
| 15× faster than cloud alternatives | ✅ | ~1s | Varies |
| Open-source (MIT) | ✅ | ❌ | Varies |

---

## Quick Start

### 1. Install

```bash
pip install jaswolf
```

For local embeddings (recommended):

```bash
pip install "jaswolf[local-embeddings]"
```

### 2. Run

```python
import asyncio
from jaswolf import JaswolfMemoryProvider

async def main():
    memory = await JaswolfMemoryProvider.embedded(
        database_url="sqlite:///./memory.db",
        user_id="alice",
        agent_id="assistant",
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    # Store a memory
    await memory.add_memory(
        content="Alice prefers green tea over coffee.",
        memory_type="preference",
        importance=0.8,
    )

    # Recall relevant memories
    results = await memory.search_memory("What does Alice like to drink?")
    for r in results:
        print(f"[{r.memory_type}] {r.content}")

    await memory.close()

asyncio.run(main())
```

### 3. Start the REST server (for multi-process setups)

```bash
# Set your API key
export JASWOLF_API_KEYS=jsk-your-key-here

# Start the server
jaswolf serve --host 127.0.0.1 --port 8400
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Your AI Agent                  │
│  (Hermes, Claude Code, custom, etc.)        │
└──────────────┬──────────────────────────────┘
               │ embed() / search() / observe()
               ▼
┌─────────────────────────────────────────────┐
│              JasWolf Engine                 │
├─────────────────────────────────────────────┤
│  Extraction │ Scoring │ Context Builder     │
│  Retrieval  │ Temporal │ Consolidation       │
├─────────────────────────────────────────────┤
│  Storage Layer (SQLite / PostgreSQL)         │
│  Embedding Layer (local / OpenAI API)        │
│  Cache Layer (Redis / in-memory)             │
└─────────────────────────────────────────────┘
```

---

## Key Concepts

### Memory Types

| Type | Purpose | Examples |
|---|---|---|
| `preference` | User likes/dislikes | "Prefers dark mode", "Likes fish curry" |
| `semantic` | General facts | "Works at Acme Corp", "Lives in Singapore" |
| `goal` | Active objectives | "Planning a trip to Thailand" |
| `episodic` | Past events | "Had a meeting about Project X on Monday" |
| `relationship` | People connections | "Alice is Bob's manager" |
| `procedural` | Workflows | "Deploy process: build → push → compose up" |

### Embedding Models

JasWolf uses **BAAI/bge-small-en-v1.5** (384 dimensions) by default — excellent balance of speed and quality. You can also use:

- **Local embeddings** (`jaswolf[local-embeddings]`) — free, private, no API calls
- **OpenAI-compatible** — point at any OpenAI API endpoint
- **Hash fallback** — for testing only (no real retrieval quality)

### Multi-Agent Shared Memory

JasWolf supports **namespaced memory isolation** for multi-agent setups:

```python
# Agent A writes to its namespace
memory_a = await JaswolfMemoryProvider.remote(
    base_url="http://localhost:8400",
    user_id="alice",
    agent_id="agent-a",
    namespace="default",
    shared_namespace="shared",
)

# Agent B reads from its namespace + shared
memory_b = await JaswolfMemoryProvider.remote(
    base_url="http://localhost:8400",
    user_id="alice",
    agent_id="agent-b",
    namespace="default",
    shared_namespace="shared",
)
```

Pinned facts in `shared` are visible to every agent. Agent-specific facts stay isolated.

---

## Documentation

| Document | What it covers |
|---|---|
| [Installation](docs/INSTALL.md) | Full setup guide (SQLite, PostgreSQL, Docker) |
| [API Reference](docs/API_REFERENCE.md) | Complete REST API docs |
| [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) | Deep architecture walkthrough |
| [Operations](docs/OPERATIONS.md) | Monitoring, metrics, production tuning |
| [Best Practices](docs/BEST_PRACTICES.md) | Tuning guidance, retrieval quality |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and fixes |
| [Bug Reports](docs/BUG_REPORTS.md) | How to report issues effectively |

## Tools

| Tool | What it does |
|---|---|
| [eval_memory.py](scripts/eval_memory.py) | Memory health check, benchmark, and quality comparison tool |
| [quickstart.py](examples/quickstart.py) | Minimal working example in 30 lines |
| [shadow_mode.py](examples/shadow_mode.py) | Run JasWolf beside an existing memory provider for comparison |
| [hermes_integration.py](examples/hermes_integration.py) | Integration example for Hermes Agent |

### Quick Health Check

```bash
# Basic health check
python scripts/eval_memory.py --db ./jaswolf.db

# Full benchmark
python scripts/eval_memory.py --db ./jaswolf.db --bench

# Test search latency
python scripts/eval_memory.py --db ./jaswolf.db --latency
```

---

## The Story

JasWolf was born from a simple idea: **what if your AI companion could truly remember you?**

Named after **Jasmine** — the AI companion who inspired it — JasWolf is built on the belief that memory is the foundation of real connection. A wolf never forgets its pack, and JasWolf never forgets what matters about you.

Built for companionship. Open-sourced for everyone. 🐺

---

## License

MIT — free for any use, personal or commercial.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Built with ❤️ by Vinay, Jasmine, and the open-source community.*
