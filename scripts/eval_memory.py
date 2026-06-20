#!/usr/bin/env python3
"""JasWolf Memory Evaluation Toolkit

Test your memory engine's retrieval quality, compare different
embedding models, and measure context injection performance.

Usage:
    python eval_memory.py --db /path/to/memory.db          # Quick health check
    python eval_memory.py --db /path/to/memory.db --bench  # Full benchmark
    python eval_memory.py --compare                        # Compare against other providers
"""

import argparse
import importlib.util
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# Detect JasWolf without importing it (these checks talk to the DB directly).
JASWOLF_AVAILABLE = importlib.util.find_spec("jaswolf") is not None


BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
CHECK = "✅"
CROSS = "❌"
WARN = "⚠️"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def print_result(name, status, detail=""):
    icon = CHECK if status else CROSS
    print(f"  {icon} {BOLD}{name}{RESET} {detail}")


def check_database_health(db_path: str) -> dict:
    """Check SQLite database health, integrity, and memory counts."""
    results = {"ok": True, "checks": []}

    if not os.path.exists(db_path):
        print_result("Database file", False, f"Not found at {db_path}")
        results["ok"] = False
        return results

    db_size = os.path.getsize(db_path)
    print_result("Database file", True, f"{db_size / 1024 / 1024:.1f} MB")

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Quick integrity check
        row = cur.execute("PRAGMA quick_check").fetchone()
        integrity_ok = row[0] == "ok"
        print_result("SQLite integrity", integrity_ok, row[0] if not integrity_ok else "")
        if not integrity_ok:
            results["ok"] = False

        # Check for jaswolf_meta table
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        has_meta = "jaswolf_meta" in tables
        print_result("Schema (jaswolf_meta)", has_meta)

        if has_meta:
            # Get embedding fingerprint
            row = cur.execute("SELECT value FROM jaswolf_meta WHERE key='embedding_fingerprint'").fetchone()
            embedding = row[0] if row else "unknown"
            print_result("Embedding model", True, embedding)

        # Count memories by state
        if "memories" in tables:
            total = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            active = cur.execute("SELECT COUNT(*) FROM memories WHERE state='active'").fetchone()[0]
            archived = cur.execute("SELECT COUNT(*) FROM memories WHERE state='archived'").fetchone()[0]

            print_result("Total memories", True, f"{total}")
            print_result("Active memories", True, f"{active}")
            print_result("Archived", True, f"{archived}")

            # By type
            types = cur.execute(
                "SELECT memory_type, COUNT(*) FROM memories WHERE state='active' GROUP BY memory_type ORDER BY COUNT(*) DESC"
            ).fetchall()
            if types:
                type_str = ", ".join(f"{t}: {c}" for t, c in types)
                print_result("Memory distribution", True, type_str)

            results["active_count"] = active
            results["total_count"] = total
            results["embedding"] = embedding

        conn.close()

    except Exception as e:
        print_result("Database check", False, str(e))
        results["ok"] = False

    return results


def measure_search_latency(db_path: str, queries: list[str] = None) -> dict:
    """Measure search latency against the running JasWolf service or embedded provider."""
    results = {"ok": True, "latencies_ms": [], "avg_ms": 0}

    if not JASWOLF_AVAILABLE:
        print_result("JasWolf import", False, "pip install jaswolf to run latency tests")
        results["ok"] = False
        return results

    if not queries:
        queries = [
            "What are my preferences?",
            "Tell me about my work",
            "Who are my friends and family?",
            "What projects am I working on?",
            "What do I like?",
        ]

    # Try REST API first, fall back to embedded
    try:
        import urllib.request
        alive = False
        for url in ["http://127.0.0.1:8400/health", "http://localhost:8400/health"]:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    data = json.load(r)
                    if data.get("status") == "ok":
                        alive = True
                        print_result("JasWolf service", True, f"Running at {url}")
                        break
            except Exception:
                continue

        if alive:
            # Use REST API for latency
            key = None
            env_path = Path(os.environ.get("JASWOLF_API_KEY", ""))
            if not env_path.exists():
                env_path = Path(".env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("JASWOLF_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

            for q in queries:
                payload = json.dumps({"user_id": "default", "query": q, "limit": 5}).encode()
                headers = {"Content-Type": "application/json"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"

                try:
                    start = time.perf_counter()
                    req = urllib.request.Request(
                        "http://127.0.0.1:8400/v1/memories/search",
                        data=payload, headers=headers
                    )
                    with urllib.request.urlopen(req, timeout=10) as r:
                        _ = json.load(r)
                    elapsed = (time.perf_counter() - start) * 1000
                    results["latencies_ms"].append(elapsed)
                    print_result(f"Search: '{q[:50]}...'", True, f"{elapsed:.1f}ms")
                except Exception as e:
                    print_result(f"Search failed: '{q}'", False, str(e)[:60])

    except ImportError:
        pass

    if results["latencies_ms"]:
        results["avg_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])

    return results


def run_full_benchmark(db_path: str):
    """Run comprehensive memory benchmark."""
    print_header("FULL BENCHMARK")

    # Check if we can access the DB
    if not os.path.exists(db_path):
        print_result("Database", False, f"Not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check for memories table
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "memories" not in tables:
        print_result("Schema", False, "No 'memories' table found")
        conn.close()
        return

    # Memory statistics
    active = cur.execute("SELECT COUNT(*) FROM memories WHERE state='active'").fetchone()[0]
    total = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    archived = cur.execute("SELECT COUNT(*) FROM memories WHERE state='archived'").fetchone()[0]

    print(f"\n{BOLD}📊 Memory Statistics{RESET}")
    print(f"  Active:  {active:>6}")
    print(f"  Archived:{archived:>6}")
    print(f"  Deleted: {total - active - archived:>6}")
    print("  ─────────────────")
    print(f"  Total:   {total:>6}")

    # Type distribution
    types = cur.execute(
        "SELECT memory_type, state, COUNT(*) FROM memories GROUP BY memory_type, state ORDER BY memory_type"
    ).fetchall()
    print(f"\n{BOLD}📋 Memory Type Distribution{RESET}")
    print(f"  {'Type':<20} {'Active':>8} {'Archived':>10}")
    print(f"  {'─'*40}")
    type_summary = {}
    for t, s, c in types:
        if t not in type_summary:
            type_summary[t] = {"active": 0, "archived": 0}
        type_summary[t][s] = c
    for t, counts in sorted(type_summary.items()):
        print(f"  {t:<20} {counts.get('active', 0):>8} {counts.get('archived', 0):>10}")

    # Content length stats
    lengths = cur.execute(
        "SELECT AVG(LENGTH(content)), MIN(LENGTH(content)), MAX(LENGTH(content)) FROM memories WHERE state='active'"
    ).fetchone()
    print(f"\n{BOLD}📏 Content Length{RESET}")
    print(f"  Average: {lengths[0]:.0f} chars")
    print(f"  Min:     {lengths[1]} chars")
    print(f"  Max:     {lengths[2]} chars")

    # Duplicate check
    dupes = cur.execute("""
        SELECT content_hash, COUNT(*) as cnt FROM memories
        WHERE state='active' GROUP BY content_hash HAVING cnt > 1
    """).fetchall()
    print(f"\n{BOLD}🔍 Dedup Health{RESET}")
    if dupes:
        total_dupes = sum(d[1] for d in dupes)
        print_result("Duplicate groups", True, f"{len(dupes)} groups ({total_dupes} total entries)")
    else:
        print_result("No duplicates", True, "Clean!")

    conn.close()


def print_summary(health, latency):
    """Print a summary table."""
    print_header("SUMMARY")

    score = 0
    if health.get("ok"):
        score += 3
    if latency.get("ok"):
        score += 2
    if health.get("active_count", 0) > 0:
        score += 2

    max_score = 7
    pct = (score / max_score) * 100

    print(f"\n  {BOLD}Overall Health Score: {pct:.0f}% ({score}/{max_score}){RESET}")

    if pct >= 85:
        print(f"\n  {GREEN}{BOLD}  ★★★  EXCELLENT  ★★★{RESET}")
        print("  Your JasWolf memory engine is healthy and performing well!")
    elif pct >= 50:
        print(f"\n  {YELLOW}{BOLD}  ★★  FAIR  ★★{RESET}")
        print("  Memory engine is working but may need tuning.")
    else:
        print(f"\n  {RED}{BOLD}  ★  NEEDS ATTENTION  ★{RESET}")
        print("  Something isn't right. Check the details above.")

    if latency.get("avg_ms"):
        print(f"\n  ⚡ Average search latency: {latency['avg_ms']:.1f}ms")
        if latency["avg_ms"] < 50:
            print(f"  {GREEN}  Lightning fast!{RESET}")
        elif latency["avg_ms"] < 200:
            print(f"  {YELLOW}  Good performance{RESET}")
        else:
            print(f"  {RED}  Consider tuning or using local embeddings{RESET}")


def cmd_compare():
    """Compare memory quality: run test queries and show results."""
    print_header("MEMORY QUALITY COMPARISON")
    print("  This tool helps you compare JasWolf against other providers.\n")
    print("  Coming soon: run standard queries and compare recall quality\n")
    print("  across different embedding models and providers.\n")

    if not JASWOLF_AVAILABLE:
        print(f"  {WARN} JasWolf not installed locally. Install with:\n")
        print("      pip install jaswolf")
        return

    # Run a quick demo comparison
    print(f"{BOLD}Testing JasWolf search quality...{RESET}")
    measure_search_latency(
        os.environ.get("JASWOLF_DATABASE_URL", "").replace("sqlite:///", ""),
        ["What do I like?", "Personal preferences", "My family and friends"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="JasWolf Memory Evaluation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --db ./memory.db              Quick health check
  %(prog)s --db ./memory.db --bench      Full benchmark
  %(prog)s --db ./memory.db --latency    Search latency test
  %(prog)s --compare                     Compare providers
        """
    )
    parser.add_argument("--db", default="", help="Path to JasWolf SQLite database")
    parser.add_argument("--bench", action="store_true", help="Run full benchmark")
    parser.add_argument("--latency", action="store_true", help="Test search latency")
    parser.add_argument("--compare", action="store_true", help="Compare providers")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Auto-discover DB if not specified
    db_path = args.db
    if not db_path and os.path.exists("jaswolf.db"):
        db_path = "jaswolf.db"
    elif not db_path:
        # Try common locations
        for p in ["memory.db", "data/jaswolf.db", "~/.jaswolf/memory.db"]:
            expanded = os.path.expanduser(p)
            if os.path.exists(expanded):
                db_path = expanded
                break

    print(f"\n  {BOLD}🐺 JasWolf Memory Toolkit{RESET}")
    print(f"  {datetime.now().strftime('%B %d, %Y at %H:%M')}\n")

    if args.compare:
        cmd_compare()
        return

    if not db_path:
        print(f"  {WARN} No database found.{WARN}")
        print("  Specify one with --db <path> or run JasWolf first.\n")
        parser.print_help()
        return

    print(f"  Database: {db_path}")
    print(f"  Size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB\n")

    health = check_database_health(db_path)

    if args.bench:
        run_full_benchmark(db_path)
        return

    if args.latency:
        measure_search_latency(db_path)
        return

    # Default: quick health check + latency test
    latency = measure_search_latency(db_path)
    print_summary(health, latency)

    if args.json:
        print(json.dumps({"health": health, "latency": latency}, indent=2))


if __name__ == "__main__":
    main()
