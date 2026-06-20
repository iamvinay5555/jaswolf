#!/usr/bin/env python3
"""Sync memories from Mem0 into JasWolf — safe to re-run (idempotent).

Pulls every memory Mem0 holds for the user and adds each to the running JasWolf
REST server. JASWOLF dedups on content hash + near-duplicate vector match, so
re-running only *adds what's missing* and reinforces the rest — it never
duplicates. Mem0 is read-only here; nothing on the Mem0 side is changed.

Run on the VPS in the **Hermes venv** (where both `mem0` and `jaswolf` are
installed):

    # 1. count only — answers "how many do we have", writes nothing to JasWolf
    python migrate_mem0.py --count

    # 2. do the sync once the count looks right
    python migrate_mem0.py --apply

Env (reuses the Mem0 plugin's + the JasWolf plugin's existing config):
    MEM0_API_KEY        set => Mem0 cloud; unset => Mem0 OSS (local)
    MEM0_USER_ID        default default
    JASWOLF_API_URL        default http://127.0.0.1:8400
    JASWOLF_API_KEY        the jaswolf-serve key
    JASWOLF_MEMORY_USER_ID default default  (target user in JasWolf)

If Mem0 auto-detection fails (custom OSS config), paste your Mem0 plugin's
client-init code to Claude and he'll adapt the `pull_mem0()` block.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


_PAGE_SIZE = 100  # Mem0 cloud's default/max page size


def _items(resp) -> list:
    """Normalize a get_all response to a list of memory items."""
    if isinstance(resp, dict):
        return resp.get("results", resp.get("memories", []))
    return resp or []


def _to_rec(it) -> dict | None:
    if isinstance(it, str):
        return {"text": it, "metadata": {}}
    text = it.get("memory") or it.get("text") or it.get("content")
    return {"text": text, "metadata": it.get("metadata") or {}} if text else None


def _cloud_get_page(client, filters: dict, page: int):
    """One page from Mem0 cloud, tolerating SDK signature differences.
    Raises TypeError if the SDK accepts no pagination kwargs at all."""
    try:
        return client.get_all(version="v2", filters=filters, page=page, page_size=_PAGE_SIZE)
    except TypeError:
        return client.get_all(filters=filters, page=page, page_size=_PAGE_SIZE)


def pull_mem0(user_id: str) -> list[dict]:
    """Return ALL Mem0 memories as [{'text':..., 'metadata':...}, ...].

    Mem0 cloud paginates get_all (default 100/page), so a single call returns
    only page 1 — we MUST loop pages or we silently under-count.
    """
    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:  # OSS / local — no cloud pagination
        from mem0 import Memory

        client = Memory()
        try:
            raw = client.get_all(filters={"user_id": user_id})
        except TypeError:
            raw = client.get_all(user_id=user_id)
        return [r for it in _items(raw) if (r := _to_rec(it))]

    from mem0 import MemoryClient

    client = MemoryClient(api_key=api_key)
    filters = {"user_id": user_id}

    # probe page 1 — if the SDK rejects pagination kwargs, fall back to one call
    try:
        first = _cloud_get_page(client, filters, 1)
    except TypeError:
        raw = client.get_all(filters=filters)
        items = _items(raw)
        if items and len(items) % _PAGE_SIZE == 0:
            print(f"  ! WARNING: got exactly {len(items)} with no pagination support — "
                  "this may be a server-side cap; verify the true total in the Mem0 dashboard.")
        return [r for it in items if (r := _to_rec(it))]

    out: list = []
    page = 1
    resp = first
    while True:
        items = _items(resp)
        out.extend(items)
        print(f"  Mem0 page {page}: {len(items)} (running total {len(out)})")
        if len(items) < _PAGE_SIZE:
            break
        page += 1
        if page > 5000:  # safety
            print("  ! 5000-page safety cap hit — stopping")
            break
        resp = _cloud_get_page(client, filters, page)
    return [r for it in out if (r := _to_rec(it))]


async def sync(apply: bool) -> int:
    mem0_user = os.environ.get("MEM0_USER_ID", "default")
    jaswolf_user = os.environ.get("JASWOLF_MEMORY_USER_ID", "default")

    memories = pull_mem0(mem0_user)
    print(f"Mem0 ({'cloud' if os.environ.get('MEM0_API_KEY') else 'OSS'}) "
          f"user={mem0_user}: {len(memories)} memories")
    if memories:
        print("  sample:", (memories[0]['text'][:100]))

    if not apply:
        print("\n--count only; nothing written. Re-run with --apply to sync.")
        return 0

    from jaswolf import JaswolfMemoryProvider

    provider = JaswolfMemoryProvider.remote(
        os.environ.get("JASWOLF_API_URL", "http://127.0.0.1:8400"),
        api_key=os.environ.get("JASWOLF_API_KEY"),
        user_id=jaswolf_user,
    )
    created = reinforced = failed = 0
    try:
        for i, mem in enumerate(memories, 1):
            try:
                res = await provider.add_memory(
                    mem["text"],
                    metadata={"source": "mem0", **(mem["metadata"] or {})},
                )
                if res.get("created"):
                    created += 1
                else:
                    reinforced += 1
            except Exception as exc:
                failed += 1
                print(f"  ! failed [{i}]: {exc}")
            if i % 50 == 0:
                print(f"  ...{i}/{len(memories)}")
    finally:
        await provider.close()

    print(
        f"\nSync complete: {created} NEW (were missing), "
        f"{reinforced} already in JasWolf (deduped), {failed} failed."
    )
    if created == 0 and failed == 0:
        print("→ JasWolf was already fully in sync with Mem0. Nothing was missing.")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync Mem0 memories into JasWolf (idempotent).")
    ap.add_argument("--apply", action="store_true", help="write to JasWolf (default: count only)")
    ap.add_argument("--count", action="store_true", help="count only (default)")
    args = ap.parse_args()
    sys.exit(asyncio.run(sync(apply=args.apply)))


if __name__ == "__main__":
    main()
