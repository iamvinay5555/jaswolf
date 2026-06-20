"""Regression tests for the 2026-06-15 live-pilot incident.

A staging `STAGING_TEST_` preference got force-pinned into every context and
dominated it, while everyday preferences also injected on unrelated turns.
Fixes under test (v0.9.0):
  - staging/test memories never enter context (provenance guard)
  - only identity-grade (importance >= context_always_pin_importance) prefs
    force-pin onto every turn; lesser prefs appear only when query-relevant
  - force-pinned set is capped by context_max_pins

context_noise_z is cranked high in these tests so the query-driven search
contributes nothing — isolating the *pinning* behavior deterministically
(the hash embedder's similarities are content-deterministic but topic-blind).
"""

from jaswolf.models import ContextRequest, MemoryCreate, MemoryType


async def _identity(service, content):
    await service.add(MemoryCreate(
        user_id="u", content=content, memory_type=MemoryType.PREFERENCE,
        importance=0.95, confidence=0.95,
    ))


async def test_staging_memory_never_in_context(service):
    service.settings.context_noise_z = 100  # isolate pinning from search
    # staging pref set at identity-grade importance — exclusion must win anyway
    await service.add(MemoryCreate(
        user_id="u", content="STAGING_TEST_20260614: Jasmine prefers Ceylon tea on Sundays",
        memory_type=MemoryType.PREFERENCE, importance=0.95, confidence=0.95,
    ))
    await _identity(service, "Do not call Alice Mr Smith or infer his surname.")

    text = (await service.build_context(
        ContextRequest(user_id="u", query="weather in Lisbon next week")
    )).text
    assert "ceylon" not in text.lower()      # staging excluded by provenance guard
    assert "Mr Smith" in text                 # real identity pin still present


async def test_metadata_test_flag_excludes(service):
    service.settings.context_noise_z = 100
    await service.add(MemoryCreate(
        user_id="u", content="Alice prefers Earl Grey", memory_type=MemoryType.PREFERENCE,
        importance=0.95, confidence=0.95, metadata={"test": True},
    ))
    await _identity(service, "Do not call Alice Mr Smith or infer his surname.")
    text = (await service.build_context(ContextRequest(user_id="u", query="weather"))).text
    assert "earl grey" not in text.lower()   # metadata.test=true excluded
    assert "Mr Smith" in text


async def test_low_importance_pref_not_force_pinned_offtopic(service):
    service.settings.context_noise_z = 100   # nothing surfaces via search
    await service.add(MemoryCreate(
        user_id="u", content="Alice prefers a window seat", memory_type=MemoryType.PREFERENCE,
        importance=0.85, confidence=0.9,     # below the 0.9 always-pin floor
    ))
    await _identity(service, "Do not call Alice Mr Smith or infer his surname.")
    text = (await service.build_context(ContextRequest(user_id="u", query="weather"))).text
    assert "window seat" not in text         # 0.85 < 0.9 -> not force-pinned
    assert "Mr Smith" in text                 # identity-grade -> force-pinned


async def test_identity_pin_survives_offtopic(service):
    service.settings.context_noise_z = 100
    await _identity(service, "Do not call Alice Mr Smith or infer his surname.")
    text = (await service.build_context(
        ContextRequest(user_id="u", query="how do rainbows form")
    )).text
    assert "Mr Smith" in text                 # safety guardrail present even off-topic


async def test_strict_mode_requires_always_pin_flag(service):
    # Jasmine strict mode: high importance alone must NOT force-pin; only the
    # explicit metadata.always_pin flag does.
    service.settings.context_noise_z = 100
    service.settings.context_pin_requires_always_pin = True
    await service.add(MemoryCreate(
        user_id="u", content="High importance but unflagged preference",
        memory_type=MemoryType.PREFERENCE, importance=0.97, confidence=0.97,
    ))
    await service.add(MemoryCreate(
        user_id="u", content="Do not call Alice Mr Smith", memory_type=MemoryType.PREFERENCE,
        importance=0.95, confidence=0.95, metadata={"always_pin": True},
    ))
    text = (await service.build_context(ContextRequest(user_id="u", query="weather"))).text
    assert "High importance but unflagged" not in text   # importance alone -> not pinned
    assert "Mr Smith" in text                              # explicit always_pin -> pinned


async def test_pin_budget_caps_force_pins(service):
    service.settings.context_noise_z = 100
    service.settings.context_max_pins = 2
    for i in range(5):
        await _identity(service, f"Identity guardrail number {i} alpha")
    text = (await service.build_context(ContextRequest(user_id="u", query="weather"))).text
    present = sum(1 for i in range(5) if f"number {i}" in text)
    assert present <= 2                       # capped by context_max_pins
