#!/usr/bin/env python3
"""One-time migration: rescope JasWolf memories for the multi-agent model.

Moves shared user facts into namespace='shared', marks identity/safety facts
always_pin, and deletes staging/test pollution — so the live bots actually
read the right memories (v0.10.0 multi-agent scoping).

SAFE BY DEFAULT: every command is a dry-run unless you pass --apply.
Run with `jaswolf-serve` STOPPED (single writer) and AFTER a backup:
    jaswolf backup --out ~/.hermes/backups/jas0/pre-rescope.db
    systemctl --user stop jaswolf-serve

Usage:
    python rescope_memories.py --db <bge.db> --list                  # review (read-only)
    python rescope_memories.py --db <bge.db> --to-shared ID1,ID2 --apply
    python rescope_memories.py --db <bge.db> --always-pin ID1,ID2 --apply
    python rescope_memories.py --db <bge.db> --delete-staging --apply
    python rescope_memories.py --db <bge.db> --archive-namespace shadow --apply
    python rescope_memories.py --db <bge.db> --restore-namespace shadow --apply  # undo the above
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

_STAGING = (
    "lower(content) LIKE '%staging_test%' OR lower(content) LIKE '%__smoke_test__%' "
    "OR lower(content) LIKE '%jaswolf_test%' "
    "OR json_extract(coalesce(metadata,'{}'),'$.test') = 1 "
    "OR json_extract(coalesce(metadata,'{}'),'$.staging') = 1"
)


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_list(conn: sqlite3.Connection) -> None:
    print("# Namespaces present (count):")
    for r in conn.execute("SELECT namespace, count(*) n FROM memories GROUP BY namespace ORDER BY n DESC"):
        print(f"    {r['namespace']}: {r['n']}")
    print("\n# Preferences / goals — candidates to move to 'shared' / mark always_pin:")
    for r in conn.execute(
        "SELECT id, namespace, memory_type, importance, confidence, substr(content,1,70) c "
        "FROM memories WHERE memory_type IN ('preference','goal') ORDER BY importance DESC LIMIT 50"
    ):
        print(f"    {r['id']}  ns={r['namespace']:<10} imp={r['importance']:.2f} "
              f"{r['memory_type']:<10} {r['c']}")
    # golden facts are often SEMANTIC, not preference/goal — scan by keyword too
    print("\n# Golden-fact keyword matches (any type — review for the shared move):")
    for kw in ("naik", "pockettts", "elevenlabs", "mrt", "harbourfront", "harborfront",
               "warp", "socks5", "telegram", "minimax", "deepseek", "sgodds", "jasx"):
        for r in conn.execute(
            "SELECT id, namespace, memory_type, substr(content,1,70) c FROM memories "
            "WHERE lower(content) LIKE ? LIMIT 5", (f"%{kw}%",)
        ):
            print(f"    [{kw}] {r['id']}  ns={r['namespace']:<9} {r['memory_type']:<10} {r['c']}")

    print("\n# Staging/test memories (delete these):")
    for r in conn.execute(f"SELECT id, substr(content,1,70) c FROM memories WHERE {_STAGING}"):
        print(f"    {r['id']}  {r['c']}")


def cmd_to_shared(conn: sqlite3.Connection, ids: list[str], apply: bool) -> None:
    for mid in ids:
        row = conn.execute("SELECT namespace, substr(content,1,60) c FROM memories WHERE id=?",
                           (mid,)).fetchone()
        if not row:
            print(f"    ! not found: {mid}")
            continue
        print(f"    {mid}: ns {row['namespace']} -> shared   {row['c']}")
        if apply:
            conn.execute("UPDATE memories SET namespace='shared' WHERE id=?", (mid,))
    if apply:
        conn.commit()


def cmd_always_pin(conn: sqlite3.Connection, ids: list[str], apply: bool) -> None:
    for mid in ids:
        row = conn.execute("SELECT metadata FROM memories WHERE id=?", (mid,)).fetchone()
        if not row:
            print(f"    ! not found: {mid}")
            continue
        md = json.loads(row["metadata"] or "{}")
        md["always_pin"] = True
        print(f"    {mid}: metadata.always_pin = true")
        if apply:
            conn.execute("UPDATE memories SET metadata=? WHERE id=?", (json.dumps(md), mid))
    if apply:
        conn.commit()


def cmd_delete_staging(conn: sqlite3.Connection, apply: bool) -> None:
    rows = conn.execute(f"SELECT id, substr(content,1,60) c FROM memories WHERE {_STAGING}").fetchall()
    print(f"    {len(rows)} staging/test memories")
    for r in rows:
        print(f"    delete {r['id']}  {r['c']}")
    if apply:
        conn.executescript("DELETE FROM memories WHERE " + _STAGING)
        conn.commit()


def cmd_archive_namespace(conn: sqlite3.Connection, namespace: str, apply: bool) -> None:
    """Archive (NOT delete) every ACTIVE row in a namespace — reversible cleanup
    for historical corpora like 'shadow' that inflate counts but never get read
    by the live bots. Idempotent: only touches active rows. Stamps a marker so
    --restore-namespace can later un-archive exactly these rows."""
    n = conn.execute(
        "SELECT count(*) n FROM memories WHERE namespace=? AND state='active'", (namespace,)
    ).fetchone()["n"]
    print(f"    {n} active memories in namespace='{namespace}' -> archived (reversible)")
    if apply:
        # mark the rows THIS sweep archived so restore is precise (won't revive
        # rows that were archived earlier for other reasons)
        conn.execute(
            "UPDATE memories SET state='archived', "
            "metadata=json_set(coalesce(metadata,'{}'), '$.ns_sweep_archived', 1) "
            "WHERE namespace=? AND state='active'",
            (namespace,),
        )
        conn.commit()


def cmd_restore_namespace(conn: sqlite3.Connection, namespace: str, apply: bool) -> None:
    """Reverse --archive-namespace: re-activate ONLY the rows a prior sweep
    archived (matched by the ns_sweep_archived marker), so rows archived earlier
    for other reasons stay archived. (Sweeps done before the marker existed are
    not matched — roll those back from the pre-archive backup instead.)"""
    where = (
        "namespace=? AND state='archived' "
        "AND json_extract(coalesce(metadata,'{}'),'$.ns_sweep_archived')=1"
    )
    n = conn.execute(f"SELECT count(*) n FROM memories WHERE {where}", (namespace,)).fetchone()["n"]
    print(f"    {n} sweep-archived memories in namespace='{namespace}' -> active")
    if n == 0:
        print("    (nothing carries the sweep marker; restore an older archive from "
              "the pre-archive backup instead)")
    if apply:
        conn.execute(
            "UPDATE memories SET state='active', "
            f"metadata=json_remove(metadata,'$.ns_sweep_archived') WHERE {where}",
            (namespace,),
        )
        conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Rescope JasWolf memories (multi-agent migration).")
    ap.add_argument("--db", required=True, help="path to the sqlite DB file")
    ap.add_argument("--list", action="store_true", help="review memories (read-only)")
    ap.add_argument("--to-shared", default="", help="comma-separated ids -> namespace=shared")
    ap.add_argument("--always-pin", default="", help="comma-separated ids -> metadata.always_pin")
    ap.add_argument("--delete-staging", action="store_true", help="delete staging/test memories")
    ap.add_argument("--archive-namespace", default="",
                    help="archive (reversible) all ACTIVE rows in this namespace, e.g. shadow")
    ap.add_argument("--restore-namespace", default="",
                    help="reverse --archive-namespace: re-activate only rows a prior sweep archived")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    if not args.apply and not args.list:
        print("(dry-run — add --apply to write; or --list to review)\n")
    conn = _connect(args.db)
    try:
        if args.list:
            cmd_list(conn)
            return
        if args.to_shared:
            cmd_to_shared(conn, [s.strip() for s in args.to_shared.split(",") if s.strip()], args.apply)
        if args.always_pin:
            cmd_always_pin(conn, [s.strip() for s in args.always_pin.split(",") if s.strip()], args.apply)
        if args.delete_staging:
            cmd_delete_staging(conn, args.apply)
        if args.archive_namespace:
            cmd_archive_namespace(conn, args.archive_namespace, args.apply)
        if args.restore_namespace:
            cmd_restore_namespace(conn, args.restore_namespace, args.apply)
        any_op = (args.to_shared or args.always_pin or args.delete_staging
                  or args.archive_namespace or args.restore_namespace)
        if not any_op:
            print("nothing to do — use --list, --to-shared, --always-pin, "
                  "--delete-staging, --archive-namespace, or --restore-namespace")
    finally:
        conn.close()
    if not args.apply and (
        args.to_shared or args.always_pin or args.delete_staging
        or args.archive_namespace or args.restore_namespace
    ):
        print("\n(dry-run only — re-run with --apply to write)")


if __name__ == "__main__":
    sys.exit(main())
