"""Shared validation helpers for model response caches."""

import os


INTERNAL_ERROR_PREFIXES = ("Error in ", "ERROR:")


def add_cache_suffix(filename: str) -> str:
    """Add the optional benchmark-run suffix before a cache extension."""
    suffix = os.environ.get("CARDBIOMEDBENCH_CACHE_SUFFIX")
    if not suffix:
        return filename
    stem, extension = os.path.splitext(filename)
    return f"{stem}_{suffix}{extension}"


def is_valid_cached_response(value) -> bool:
    """Return whether a value is a non-empty model response, not an error marker."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.startswith(INTERNAL_ERROR_PREFIXES)
    )


def filter_valid_cache(cache: dict) -> dict:
    """Return a cache containing only reusable model responses."""
    if not isinstance(cache, dict):
        raise ValueError("Response cache is not a JSON object")
    return {
        key: value
        for key, value in cache.items()
        if is_valid_cached_response(value)
    }
