# Installing and Configuring JasWolf

JasWolf runs wherever Python runs — Linux, macOS, and Windows. This guide
covers every installation method, from a quick local setup to a production
deployment, and how to wire it into Hermes Agent as your memory provider.

---

## Table of Contents

1. [Quick Start (All Platforms)](#1-quick-start-all-platforms)
2. [Linux Installation](#2-linux-installation)
3. [macOS Installation](#3-macos-installation)
4. [Windows Installation](#4-windows-installation)
5. [Docker Deployment](#5-docker-deployment)
6. [Wiring into Hermes Agent](#6-wiring-into-hermes-agent)
7. [Verifying the Installation](#7-verifying-the-installation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick Start (All Platforms)

### Prerequisites

- **Python 3.11 or newer**
- **Git** (to clone the repository)
- **pip** (Python package manager)

### Install from source (recommended)

```bash
# Clone the repository
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf

# Install the package and its dependencies
pip install -e .

# For local embeddings (recommended for best performance):
pip install -e ".[local-embeddings]"
```

### Install directly from GitHub (no clone needed)

```bash
pip install "jaswolf @ git+https://github.com/iamvinay5555/jaswolf.git"

# With local embeddings:
pip install "jaswolf[local-embeddings] @ git+https://github.com/iamvinay5555/jaswolf.git"
```

### Start the server

```bash
# Set an API key (required for production)
export JASWOLF_API_KEYS=jsk-your-key-here

# Start the REST server
jaswolf serve --host 127.0.0.1 --port 8400
```

That's it. Your JasWolf memory server is now running on `http://127.0.0.1:8400`.

---

## 2. Linux Installation

### 2a. Quick local setup (personal workstation)

```bash
# Install Python 3.11+ if not already installed
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# Clone and install
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local-embeddings]"

# Generate an API key
python3 -c "import secrets; print('jsk-' + secrets.token_hex(32))"

# Start the server
JASWOLF_API_KEYS=jsk-your-generated-key jaswolf serve --host 127.0.0.1 --port 8400
```

### 2b. Production deployment with systemd (server/VPS)

This is the recommended setup for a production JasWolf server that runs
as a background service and starts automatically on boot.

#### Step 1: Clone and install

```bash
# Install as a dedicated user (recommended)
sudo useradd -r -s /bin/false jaswolf
sudo mkdir -p /opt/jaswolf
sudo chown jaswolf:jaswolf /opt/jaswolf

# Clone the repo
sudo git clone https://github.com/iamvinay5555/jaswolf.git /opt/jaswolf
cd /opt/jaswolf

# Create virtualenv and install
sudo -u jaswolf python3 -m venv .venv
sudo -u jaswolf .venv/bin/pip install -e ".[local-embeddings]"
```

#### Step 2: Create env file

```bash
sudo tee /opt/jaswolf/.env > /dev/null << 'EOF'
JASWOLF_API_KEYS=jsk-your-generated-key-here
JASWOLF_DATABASE_URL=sqlite:////opt/jaswolf/data/jaswolf.db
JASWOLF_EMBEDDING_PROVIDER=local
JASWOLF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
JASWOLF_LOG_LEVEL=INFO
EOF

# Create data directory
sudo mkdir -p /opt/jaswolf/data
sudo chown jaswolf:jaswolf /opt/jaswolf/data
```

#### Step 3: Install systemd service

The repo includes a pre-made systemd unit file:

```bash
sudo cp deploy/jaswolf-serve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jaswolf-serve
```

Or create it manually:

```ini
[Unit]
Description=JasWolf memory server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jaswolf
WorkingDirectory=/opt/jaswolf
EnvironmentFile=/opt/jaswolf/.env
ExecStart=/opt/jaswolf/.venv/bin/jaswolf serve --host 127.0.0.1 --port 8400
Restart=always
RestartSec=3
TimeoutStartSec=180
MemoryMax=2G
NoNewPrivileges=true

[Install]
WantedBy=default.target
```

#### Step 4: Verify the service

```bash
sudo systemctl status jaswolf-serve
curl http://127.0.0.1:8400/health
```

Expected response:
```json
{"status":"ok","uptime_seconds":...,"storage":{"backend":"sqlite","ok":true,"integrity":"ok"},...}
```

### 2c. Docker on Linux

See [Section 5: Docker Deployment](#5-docker-deployment).

---

## 3. macOS Installation

### 3a. Quick local setup

```bash
# Install Python 3.11+ (via Homebrew if needed)
brew install python@3.11 git

# Clone and install
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local-embeddings]"

# Start the server
JASWOLF_API_KEYS=jsk-your-key jaswolf serve --host 127.0.0.1 --port 8400
```

### 3b. Running as a background service (launchd)

Create a launchd plist at `~/Library/LaunchAgents/com.jaswolf.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jaswolf.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/jaswolf</string>
        <string>serve</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8400</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/opt/jaswolf</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>JASWOLF_API_KEYS</key>
        <string>jsk-your-key-here</string>
        <key>JASWOLF_DATABASE_URL</key>
        <string>sqlite:////opt/jaswolf/data/jaswolf.db</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/opt/jaswolf/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/opt/jaswolf/logs/stderr.log</string>
</dict>
</plist>
```

Then load it:

```bash
mkdir -p /opt/jaswolf/logs
launchctl load ~/Library/LaunchAgents/com.jaswolf.server.plist
```

### 3c. Docker on macOS

See [Section 5: Docker Deployment](#5-docker-deployment).

---

## 4. Windows Installation

### 4a. Using Python directly (recommended for Hermes Desktop)

#### Step 1: Install prerequisites

1. **Install Python 3.11+** from [python.org](https://www.python.org/downloads/)
   - ✅ Check **"Add Python to PATH"** during installation
   - ✅ Enable **"Install for all users"** if on a shared machine

2. **Install Git** from [git-scm.com](https://git-scm.com/download/win)
   - Default options are fine

#### Step 2: Install JasWolf

```powershell
# Open PowerShell or Command Prompt

# Clone the repository
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install JasWolf
pip install -e ".[local-embeddings]"
```

> **Note:** If `pip install jaswolf[local-embeddings]` takes too long or you
> just want a quick test, skip the local embeddings and use the hash fallback
> instead. The hash embedder is instant and has no downloads — just lower
> retrieval quality. Upgrade to real embeddings later:
> ```powershell
> pip install -e .         # no embeddings (uses hash fallback)
> # Later, upgrade:
> pip install -e ".[local-embeddings]"
> ```

#### Step 3: Start the server

```powershell
# Generate an API key
python -c "import secrets; print('jsk-' + secrets.token_hex(32))"

# Quick test (no auth required, dev mode):
$env:JASWOLF_DEV_OPEN_MODE="true"
jaswolf serve --host 127.0.0.1 --port 8400

# Production (with API key):
$env:JASWOLF_API_KEYS="jsk-your-generated-key"
jaswolf serve --host 127.0.0.1 --port 8400
```

#### Step 4: Run as a background service (optional)

For a persistent setup, create a PowerShell script `start-jaswolf.ps1`:

```powershell
# start-jaswolf.ps1
$env:JASWOLF_API_KEYS = "jsk-your-generated-key"
$env:JASWOLF_DATABASE_URL = "sqlite:///C:/Users/$env:USERNAME/.jaswolf/data/jaswolf.db"
$env:JASWOLF_EMBEDDING_PROVIDER = "hash"   # quick start; change to "local" later

# Create data directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.jaswolf\data" | Out-Null

# Start the server
jaswolf serve --host 127.0.0.1 --port 8400
```

Then use **Task Scheduler** to run this script at startup:

1. Open **Task Scheduler**
2. Click **Create Task**
3. **General tab**: Name `JasWolf Server`, check "Run whether user is logged on or not"
4. **Triggers tab**: New → "At startup"
5. **Actions tab**: New → 
   - Action: `Start a program`
   - Program: `powershell.exe`
   - Arguments: `-WindowStyle Hidden -File "C:\path\to\start-jaswolf.ps1"`

#### Step 5: Test the server

```powershell
curl http://127.0.0.1:8400/health
```

Expected response:
```json
{"status":"ok","uptime_seconds":...,"storage":{"backend":"sqlite","ok":true,"integrity":"ok"},...}
```

### 4b. Using Docker on Windows

See [Section 5: Docker Deployment](#5-docker-deployment). This is the
simplest option on Windows if you have Docker Desktop installed.

---

## 5. Docker Deployment

JasWolf includes a Docker Compose setup that works on all platforms
(Linux, macOS, Windows).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+)

### Quick start with Docker

```bash
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf

# Set your API key
export JASWOLF_API_KEYS=jsk-your-key

# Start with SQLite (default)
docker compose -f docker/docker-compose.yml up -d

# Start with PostgreSQL (for production)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.postgres.yml up -d
```

### Verify

```bash
curl http://localhost:8400/health
```

### Docker Compose configuration

The default `docker/docker-compose.yml` includes:

```yaml
services:
  jaswolf:
    build: ..
    ports:
      - "8400:8400"
    environment:
      - JASWOLF_API_KEYS=${JASWOLF_API_KEYS}
    volumes:
      - jaswolf-data:/data
```

---

## 6. Wiring into Hermes Agent

Once JasWolf is running, you can configure Hermes to use it as its
memory provider.

> **⚠️ Plugin path depends on your Hermes installation type.**
> Read the section below that matches your setup. Using the wrong path
> is the #1 cause of "memory provider not found" errors.

---

### 6a. Hermes installed from source / pip (Linux, macOS, CLI)

If you installed Hermes via `pip install hermes-agent` or from the
GitHub source, plugins live in your user home under `~/.hermes/`.

#### Step 1: Install the plugin

The plugin lives in the JasWolf repo at `integrations/hermes/jaswolf/`.

```bash
# Copy the plugin into Hermes' plugin directory
cp -r integrations/hermes/jaswolf ~/.hermes/plugins/memory/jaswolf/
```

> **Why `memory/` is needed:** Hermes scans for memory plugins at
> `~/.hermes/plugins/memory/<name>/`. The `memory/` subdirectory tells
> Hermes this is a memory provider (not a browser or model-provider plugin).

#### Step 2: Configure Hermes

Edit `~/.hermes/config.yaml`:

```yaml
memory:
  provider: jaswolf
```

#### Step 3: Set environment variables

Add to `~/.hermes/.env`:

```bash
# JasWolf server address
JASWOLF_API_URL=http://127.0.0.1:8400

# API key (must match the server's JASWOLF_API_KEYS)
JASWOLF_API_KEY=jsk-your-key

# User / agent identity
JASWOLF_MEMORY_USER_ID=default
JASWOLF_MEMORY_AGENT_ID=hermes
JASWOLF_MEMORY_NAMESPACE=default
JASWOLF_MEMORY_SHARED_NAMESPACE=shared

# Timeout (3-8s recommended — JasWolf is fast)
JASWOLF_MEMORY_TIMEOUT=8.0

# Optional: journal path for crash-proof writes
JASWOLF_MEMORY_JOURNAL=/path/to/jaswolf_journal.jsonl
```

---

### 6b. Hermes Desktop app (Windows, macOS GUI)

If you use the **Hermes Desktop app** (electron-based GUI), the plugin
path is **inside the Hermes installation directory**, not in your home
folder.

#### Step 1: Find your Hermes installation directory

The default paths are:

| Platform | Hermes install path |
|---|---|
| **Windows** | `C:\Users\<you>\AppData\Local\hermes\hermes-agent\` |
| **macOS** | `~/Library/Application Support/hermes/hermes-agent/` |
| **Linux** (AppImage) | `~/.local/share/hermes/hermes-agent/` |

To confirm, look for the `plugins/memory/` folder inside that directory:

```powershell
# Windows PowerShell
ls "C:\Users\$env:USERNAME\AppData\Local\hermes\hermes-agent\plugins\memory\"
```

```bash
# macOS / Linux
ls ~/Library/Application\ Support/hermes/hermes-agent/plugins/memory/
```

You should see built-in providers like `honcho`, `mem0`, `hindsight`, etc.
JasWolf will sit alongside them.

#### Step 2: Install the plugin

Copy the JasWolf plugin into that `plugins/memory/` directory:

```powershell
# Windows PowerShell (run as administrator if needed)
Copy-Item -Recurse integrations/hermes/jaswolf `
  "C:\Users\$env:USERNAME\AppData\Local\hermes\hermes-agent\plugins\memory\jaswolf\"
```

```bash
# macOS / Linux
cp -r integrations/hermes/jaswolf \
  ~/Library/Application\ Support/hermes/hermes-agent/plugins/memory/jaswolf/
```

> **⚠️ Important:** The plugin folder MUST be named `jaswolf` (not `JasWolf`
> or `jaswolf-memory`). This is the name Hermes uses to look up the provider
> in `memory.provider` config.

#### Step 3: Install the SDK into Hermes' Python environment

The desktop Hermes has its own bundled Python environment. You need to
install the JasWolf SDK into that specific environment, not your system
Python:

```powershell
# Windows PowerShell — find and activate Hermes' Python
$hermesPython = "C:\Users\$env:USERNAME\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"

# Ensure pip is available
& $hermesPython -m ensurepip --upgrade

# Install JasWolf SDK from your local clone
cd C:\path\to\jaswolf
& $hermesPython -m pip install -e .
```

```bash
# macOS / Linux
~/Library/Application\ Support/hermes/hermes-agent/venv/bin/python -m pip install -e /path/to/jaswolf
```

#### Step 4: Configure Hermes

The config is typically at `~/.hermes/config.yaml` or sometimes inside
the install directory. Set:

```yaml
memory:
  provider: jaswolf
```

#### Step 5: Set environment variables

On desktop Hermes, env vars go into `~/.hermes/.env`:

```bash
JASWOLF_API_URL=http://127.0.0.1:8400
JASWOLF_API_KEY=jsk-your-key
JASWOLF_MEMORY_USER_ID=default
JASWOLF_MEMORY_AGENT_ID=hermes
JASWOLF_MEMORY_NAMESPACE=default
JASWOLF_MEMORY_SHARED_NAMESPACE=shared
JASWOLF_MEMORY_TIMEOUT=8.0
```

---

### 6c. Verify and activate (all installations)

> **Windows tips:**
> - **Database path:** Use `sqlite:///C:/Users/vinay/.jaswolf/data/jaswolf.db`
>   (3 slashes + drive letter). Do NOT use `sqlite:////c/Users/...` (4 slashes)
>   — that converts to a relative path on Windows and SQLite will fail.
> - **Embeddings:** Start with `hash` to avoid downloading the ~100MB
>   sentence-transformers model. Set `JASWOLF_EMBEDDING_PROVIDER=hash` in
>   your env. Upgrade to `local` later for better retrieval quality.

#### Verify the plugin is detected

```bash
hermes memory status
```

If everything is correct, you should see:

```
  Built-in:  always active
  Provider:  jaswolf

  Plugin:    installed ✓
  Status:    available ✓

  Installed plugins:
    • jaswolf  (no setup needed) ← active
```

If you see `Provider: builtin` and no `jaswolf` in the list:

| Symptom | Likely cause | Fix |
|---|---|---|
| Plugin not listed at all | Wrong plugin path | Check 6a vs 6b above |
| Plugin listed but `NOT installed` | SDK not in Hermes' venv | Run Step 3 again |
| "Name collision" errors | Plugin folder named same as SDK | This is expected and handled |
| Still shows `builtin` | Need to restart session | Run `hermes --resume` |

#### Restart Hermes

Memory providers load at session startup. Start a fresh session:

```bash
hermes --resume
```

Then verify again:

```bash
hermes memory status
```

---

## 7. Verifying the Installation

### Health check

```bash
curl http://127.0.0.1:8400/health
```

Expected:
```json
{"status":"ok","uptime_seconds":...,"storage":{"backend":"sqlite","ok":true,"integrity":"ok"},"embeddings":{...}}
```

### Using the evaluation toolkit

The repo includes a memory health evaluation tool:

```bash
# Basic health check
python scripts/eval_memory.py --db ./jaswolf.db

# Full benchmark
python scripts/eval_memory.py --db ./jaswolf.db --bench

# Search latency test
python scripts/eval_memory.py --db ./jaswolf.db --latency
```

### Python quick test

```python
import asyncio
from jaswolf import JaswolfMemoryProvider

async def test():
    # .remote(...) is synchronous — do not await it; only its methods are async.
    memory = JaswolfMemoryProvider.remote(
        base_url="http://127.0.0.1:8400",
        user_id="test",
        agent_id="test-agent",
    )

    # Store a test memory
    await memory.add_memory(
        content="This is a test memory.",
        memory_type="semantic",
        importance=0.5,
    )

    # Recall
    results = await memory.search_memory("test memory")
    print(f"Found {len(results)} results")

    # Health
    health = await memory.health_check()
    print(f"Health: {health['status']}")

    await memory.close()

asyncio.run(test())
```

---

## 8. Troubleshooting

### "ModuleNotFoundError: No module named 'jaswolf'"

The package isn't installed in your Python environment.

**Fix:**
```bash
pip install -e /path/to/jaswolf
```

Make sure you're in the correct virtual environment if using one.

### Server won't start: "Refusing to start without authentication"

You must set `JASWOLF_API_KEYS` or enable dev mode.

**Fix:**
```bash
# Set an API key
export JASWOLF_API_KEYS=jsk-your-key

# Or for local development only (NOT for production):
export JASWOLF_DEV_OPEN_MODE=true
```

### Hermes says "Plugin not available" after copying

The plugin directory structure must match what Hermes expects.

**Fix:**
```bash
# Correct structure:
ls ~/.hermes/plugins/memory/jaswolf/
# Should show: __init__.py  plugin.yaml
```

### "jaswolf-serve.service not found" on Windows

The systemd service file is for Linux only. On Windows, use one of:
- **PowerShell background job** (see Windows section above)
- **Docker Desktop** (see Docker section)
- **Task Scheduler** (see Windows section above)

### "Address already in use" when starting the server

Another process is already using port 8400.

**Fix:**
```bash
# Find what's using the port
sudo lsof -i :8400   # Linux / macOS
netstat -ano | findstr :8400   # Windows

# Use a different port
jaswolf serve --host 127.0.0.1 --port 8401
```

### Memory writes succeed but search returns nothing

The embedding model may not have finished warming up, or you're using
the hash fallback (which has poor retrieval quality).

**Fix:**
```bash
# Ensure you installed local embeddings
pip install "jaswolf[local-embeddings]"

# Check what embedding provider is active
curl http://127.0.0.1:8400/health | python3 -m json.tool
# Look for: "fallback": false
```

### Need more help?

- Open an issue at [github.com/iamvinay5555/jaswolf/issues](https://github.com/iamvinay5555/jaswolf/issues)
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more detailed scenarios
- Review [OPERATIONS.md](OPERATIONS.md) for production tuning
