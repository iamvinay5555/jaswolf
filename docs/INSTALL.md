# Installing JASWOLF

## Requirements

* Python 3.11+ (3.12 recommended)
* Optional: PostgreSQL 15+ with pgvector, Redis 6+

## Local (embedded / development)

```bash
git clone <repo> jaswolf && cd jas0
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                      # verify: all green
.venv/bin/python examples/quickstart.py
```

That's a fully working engine: SQLite storage, in-process cache, hashing
embedder. For semantic-quality retrieval add a real embedder:

```bash
pip install ".[local-embeddings]"     # sentence-transformers + bge-small (CPU fine)
# or point at any OpenAI-compatible endpoint:
export JASWOLF_EMBEDDING_PROVIDER=openai
export JASWOLF_OPENAI_API_KEY=sk-...
```

### CPU-only servers: avoid the CUDA wheel stack

On Linux, `pip install ".[local-embeddings]"` pulls the **default torch
wheel, which bundles CUDA/NVIDIA libraries** — several GB you don't want on
a CPU-only VPS, and large enough that downloads can fail mid-stream.
Install CPU torch first, then the rest resolves against it:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
pip install "sentence-transformers>=2.5"   # torch already satisfied -> no CUDA pull
pip install .                              # jaswolf core deps are lightweight
```

Verify no GPU stack sneaked in and the model loads:

```bash
pip list | grep -Ei "nvidia|cuda|triton" || echo "clean"
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('BAAI/bge-small-en-v1.5'); print(m.get_sentence_embedding_dimension())"
```

(Found the hard way on the Hermes VPS, 2026-06-11 — see
`jasmine_feedback.md`.)

## Docker Compose (production single-VPS)

```bash
cp .env.example .env       # set POSTGRES_PASSWORD and JASWOLF_API_KEYS
docker compose -f docker/docker-compose.yml up -d
curl localhost:8400/health
```

Brings up JASWOLF + Postgres (pgvector) + Redis. Migrations apply automatically
on startup. Add Prometheus + Grafana with:

```bash
docker compose -f docker/docker-compose.yml --profile monitoring up -d
```

## Bare VPS (systemd)

```bash
pip install "jaswolf[postgres,redis,metrics]"
export JASWOLF_DATABASE_URL=postgresql://jas0:pw@localhost:5432/jaswolf
export JASWOLF_API_KEYS=your-key:hermes
jaswolf serve --host 0.0.0.0 --port 8400 --workers 2
```

`/etc/systemd/system/jaswolf.service`:

```ini
[Unit]
Description=JASWOLF memory engine
After=network.target postgresql.service

[Service]
EnvironmentFile=/etc/jaswolf/env
ExecStart=/opt/jaswolf/.venv/bin/jaswolf serve --host 0.0.0.0 --port 8400
Restart=always
User=jaswolf

[Install]
WantedBy=multi-user.target
```

## Choosing a database

| Scale | Setting |
| --- | --- |
| dev / tests | `JASWOLF_DATABASE_URL=sqlite:///./jaswolf.db` (default) |
| ≤ ~50k memories, one process | SQLite is fine in production too |
| beyond, or multi-process | `postgresql://user:pw@host:5432/jaswolf` |

The embedding dimension is fixed at first migration (`JASWOLF_EMBEDDING_DIM`,
default 384 = bge-small). Changing models with a different dimension means
re-embedding — pick the model before going to production.
