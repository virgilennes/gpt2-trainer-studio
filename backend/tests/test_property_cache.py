"""Property test: Cache idempotence (Property 11).

**Validates: Requirements 10.1, 1.1**

For any resource (model or dataset), calling is_cached and get_cache_path
twice produces equivalent results — the cache lookup is idempotent.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from backend.app.engines.cache_manager import CacheManager

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Resource names: simple alphanumeric identifiers, optionally with a slash
# to mimic org/model style names.  We keep them short and printable.
_simple_name = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_-]{0,30}", fullmatch=True)
_org_name = st.tuples(_simple_name, _simple_name).map(lambda t: f"{t[0]}/{t[1]}")
resource_names = st.one_of(_simple_name, _org_name)

# Dataset-like names that trigger the dataset heuristic
dataset_names = st.sampled_from([
    "wikitext",
    "wikitext-2-raw-v1",
    "my-dataset",
    "wiki-corpus",
])


class TestCacheIdempotence:
    """Property 11 — cache idempotence.

    Calling ``is_cached`` and ``get_cache_path`` twice for the same
    resource must produce identical results each time.
    """

    @given(resource=resource_names)
    @settings(max_examples=100)
    def test_get_cache_path_is_idempotent(self, resource: str) -> None:
        """**Validates: Requirements 10.1, 1.1**

        get_cache_path(resource) returns the same Path on every call.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cm = CacheManager(cache_root=Path(tmp))
            first = cm.get_cache_path(resource)
            second = cm.get_cache_path(resource)
            assert first == second

    @given(resource=resource_names)
    @settings(max_examples=100)
    def test_is_cached_is_idempotent_when_not_cached(self, resource: str) -> None:
        """**Validates: Requirements 10.1, 1.1**

        is_cached(resource) returns the same boolean on consecutive calls
        when the resource has NOT been cached.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cm = CacheManager(cache_root=Path(tmp))
            first = cm.is_cached(resource)
            second = cm.is_cached(resource)
            assert first == second
            assert first is False  # nothing cached yet

    @given(resource=st.one_of(resource_names, dataset_names))
    @settings(max_examples=100)
    def test_is_cached_is_idempotent_when_cached(self, resource: str) -> None:
        """**Validates: Requirements 10.1, 1.1**

        is_cached(resource) returns True consistently once the resource
        directory exists and contains at least one file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cm = CacheManager(cache_root=Path(tmp))
            # Simulate a cached resource by creating the directory with a file
            cache_path = cm.get_cache_path(resource)
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "config.json").write_text("{}")

            first = cm.is_cached(resource)
            second = cm.is_cached(resource)
            assert first == second
            assert first is True

    @given(resource=st.one_of(resource_names, dataset_names))
    @settings(max_examples=100)
    def test_cache_path_and_is_cached_consistent(self, resource: str) -> None:
        """**Validates: Requirements 10.1, 1.1**

        After populating the path returned by get_cache_path, is_cached
        reports True — and both calls are individually idempotent.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cm = CacheManager(cache_root=Path(tmp))

            path1 = cm.get_cache_path(resource)
            path2 = cm.get_cache_path(resource)
            assert path1 == path2

            # Before populating
            assert cm.is_cached(resource) is False

            # Populate
            path1.mkdir(parents=True, exist_ok=True)
            (path1 / "model.bin").write_text("data")

            # After populating — idempotent True
            assert cm.is_cached(resource) is True
            assert cm.is_cached(resource) is True
