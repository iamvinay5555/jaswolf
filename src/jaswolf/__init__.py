"""JASWOLF — long-term memory engine for autonomous agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "JaswolfSettings",
    "MemoryService",
    "JaswolfClient",
    "AsyncJaswolfClient",
    "JaswolfError",
    "JaswolfMemoryProvider",
    "Memory",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryType",
    "MemoryState",
    "SearchMode",
    "SearchQuery",
    "ContextRequest",
    "ChatMessage",
]

if TYPE_CHECKING:
    from .config import JaswolfSettings
    from .models import (
        ChatMessage,
        ContextRequest,
        Memory,
        MemoryCreate,
        MemoryState,
        MemoryType,
        MemoryUpdate,
        SearchMode,
        SearchQuery,
    )
    from .providers.hermes import JaswolfMemoryProvider
    from .sdk.client import AsyncJaswolfClient, JaswolfClient, JaswolfError
    from .service import MemoryService

_LAZY = {
    "JaswolfSettings": ("jaswolf.config", "JaswolfSettings"),
    "MemoryService": ("jaswolf.service", "MemoryService"),
    "JaswolfClient": ("jaswolf.sdk.client", "JaswolfClient"),
    "AsyncJaswolfClient": ("jaswolf.sdk.client", "AsyncJaswolfClient"),
    "JaswolfError": ("jaswolf.sdk.client", "JaswolfError"),
    "JaswolfMemoryProvider": ("jaswolf.providers.hermes", "JaswolfMemoryProvider"),
    "Memory": ("jaswolf.models", "Memory"),
    "MemoryCreate": ("jaswolf.models", "MemoryCreate"),
    "MemoryUpdate": ("jaswolf.models", "MemoryUpdate"),
    "MemoryType": ("jaswolf.models", "MemoryType"),
    "MemoryState": ("jaswolf.models", "MemoryState"),
    "SearchMode": ("jaswolf.models", "SearchMode"),
    "SearchQuery": ("jaswolf.models", "SearchQuery"),
    "ContextRequest": ("jaswolf.models", "ContextRequest"),
    "ChatMessage": ("jaswolf.models", "ChatMessage"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'jaswolf' has no attribute {name!r}")
