"""Shared validation helpers for model response caches."""


INTERNAL_ERROR_PREFIXES = ("Error in ", "ERROR:")


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
