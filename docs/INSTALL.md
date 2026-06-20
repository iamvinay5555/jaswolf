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

#### Step 3: Start the server

```powershell
# Generate an API key
python -c "import secrets; print('jsk-' + secrets.token_hex(32))"

# Start the server
$env:JASWOLF_API_KEYS = "jsk-your-generated-key"
jaswolf serve --host 127.0.0.1 --port 8400
```

#### Step 4: Run as a background service (optional)

For a persistent setup, create a PowerShell script `start-jaswolf.ps1`:

```powershell
# start-jaswolf.ps1
$env:JASWOLF_API_KEYS = "jsk-your-generated-key"
$env:JASWOLF_DATABASE_URL = "sqlite:///$env:USERPROFILE\.jaswolf\data\jaswolf.db"

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

### Step 1: Install the Hermes plugin

The plugin lives in the JasWolf repo at `integrations/hermes/jaswolf/`.
Copy it to your Hermes plugins directory:

```bash
# Linux / macOS
cp -r integrations/hermes/jaswolf ~/.hermes/plugins/memory/jaswolf/

# Windows PowerShell
Copy-Item -Recurse integrations/hermes/jaswolf $env:USERPROFILE\.hermes\plugins\memory\jaswolf\
```

### Step 2: Configure Hermes

Edit `~/.hermes/config.yaml` (or `%USERPROFILE%\.hermes\config.yaml` on Windows):

```yaml
memory:
  provider: jaswolf
```

### Step 3: Set environment variables

Add to `~/.hermes/.env` (or the Hermes process environment):

```bash
# The URL of your running JasWolf server
JASWOLF_API_URL=http://127.0.0.1:8400

# API key (must match the server's JASWOLF_API_KEYS)
JASWOLF_API_KEY=jsk-your-key

# User identifier (for multi-user setups)
JASWOLF_MEMORY_USER_ID=default

# Agent identifier (for multi-agent shared memory)
JASWOLF_MEMORY_AGENT_ID=hermes

# Memory namespace
JASWOLF_MEMORY_NAMESPACE=default

# Shared namespace (for cross-agent memory)
JASWOLF_MEMORY_SHARED_NAMESPACE=shared

# Timeout in seconds (JasWolf is fast — 3-8s is fine)
JASWOLF_MEMORY_TIMEOUT=8.0

# Optional: journal path for crash-proof writes
JASWOLF_MEMORY_JOURNAL=/path/to/jaswolf_journal.jsonl
```

### Step 4: Verify the integration

```bash
hermes memory status
```

Expected output:
```
  Built-in:  always active
  Provider:  jaswolf

  Plugin:    installed ✓
  Status:    available ✓

  Installed plugins:
    • jaswolf  (no setup needed) ← active
```

### Step 5: Restart Hermes

Memory providers are loaded at startup. Restart your Hermes session:

```bash
# If using the gateway:
hermes gateway restart

# If using the CLI, start a new session:
hermes --resume
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
    memory = await JaswolfMemoryProvider.remote(
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
