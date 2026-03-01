"""In-memory cache for per-page markdown extractions.

Keyed by (document_id, page_num). Shared between the MCP server and the
FastAPI server so cache-bust API calls take effect immediately.

Cache is lost on process restart — use the /cache endpoints to bust manually
when switching LLMs or after re-ingesting a document.
"""
from __future__ import annotations

_cache: dict[tuple[str, int], str] = {}


def get(doc_id: str, page_num: int) -> str | None:
    return _cache.get((doc_id, page_num))


def put(doc_id: str, page_num: int, content: str) -> None:
    _cache[(doc_id, page_num)] = content


def invalidate(doc_id: str | None = None) -> int:
    """Clear cache entries. Pass doc_id to clear one document, or None for all.

    Returns the number of entries cleared.
    """
    global _cache
    if doc_id is None:
        count = len(_cache)
        _cache = {}
        return count
    keys = [k for k in _cache if k[0] == doc_id]
    for k in keys:
        del _cache[k]
    return len(keys)


def size() -> int:
    return len(_cache)
