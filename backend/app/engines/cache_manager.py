"""Cache Manager wrapping Hugging Face cache directory management.

Tracks checkpoint locations for resume capability and provides
cache lookup for models and datasets.

**Validates: Requirements 10.1, 10.2**
"""

from __future__ import annotations

import os
from pathlib import Path


# Default HF cache directory (respects HF_HOME / TRANSFORMERS_CACHE env vars)
_DEFAULT_HF_CACHE = Path(
    os.environ.get(
        "HF_HOME",
        os.environ.get(
            "TRANSFORMERS_CACHE",
            Path.home() / ".cache" / "huggingface",
        ),
    )
)

# Subdirectory inside HF cache where hub models/datasets live
_HUB_SUBDIR = "hub"
_DATASETS_SUBDIR = "datasets"


class CacheManager:
    """Wraps Hugging Face cache directory management.

    Provides helpers to check whether a model or dataset is already
    cached locally and to resolve the on-disk path for a given resource.
    Also manages a checkpoint directory for training resume capability.
    """

    def __init__(self, cache_root: Path | None = None, checkpoint_root: Path | None = None) -> None:
        self._cache_root = Path(cache_root) if cache_root else _DEFAULT_HF_CACHE
        self._checkpoint_root = (
            Path(checkpoint_root) if checkpoint_root else self._cache_root / "checkpoints"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def cache_root(self) -> Path:
        """Return the root cache directory."""
        return self._cache_root

    @property
    def checkpoint_root(self) -> Path:
        """Return the root checkpoint directory."""
        return self._checkpoint_root

    def is_cached(self, resource: str) -> bool:
        """Check whether *resource* (model or dataset name) is already cached.

        The lookup checks both the ``hub`` and ``datasets`` subdirectories
        inside the cache root.  A resource is considered cached when its
        directory exists **and** contains at least one file.

        Parameters
        ----------
        resource:
            A Hugging Face resource identifier, e.g. ``"gpt2"`` or
            ``"wikitext"``.  Slash-separated names (``"org/model"``) are
            normalised to ``"models--org--model"`` following HF convention.
        """
        path = self.get_cache_path(resource)
        if path.is_dir() and any(path.iterdir()):
            return True
        return False

    def get_cache_path(self, resource: str) -> Path:
        """Return the expected cache directory for *resource*.

        The path follows Hugging Face Hub conventions:
        ``<cache_root>/hub/models--<resource>`` for models and
        ``<cache_root>/datasets/<resource>`` for datasets.

        If the resource looks like a dataset name (contains ``"wiki"`` or
        ``"dataset"`` case-insensitively), the datasets subdirectory is
        used; otherwise the hub/models subdirectory is used.
        """
        normalised = resource.replace("/", "--")
        if self._looks_like_dataset(resource):
            return self._cache_root / _DATASETS_SUBDIR / normalised
        return self._cache_root / _HUB_SUBDIR / f"models--{normalised}"

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def get_checkpoint_dir(self, run_name: str, epoch: int) -> Path:
        """Return the checkpoint directory for a given run and epoch.

        Creates the directory if it does not exist.
        """
        ckpt_dir = self._checkpoint_root / run_name / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def list_checkpoints(self, run_name: str) -> list[Path]:
        """List all checkpoint directories for *run_name*, sorted by epoch."""
        run_dir = self._checkpoint_root / run_name
        if not run_dir.is_dir():
            return []
        return sorted(
            (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-epoch-")),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )

    def latest_checkpoint(self, run_name: str) -> Path | None:
        """Return the most recent checkpoint for *run_name*, or ``None``."""
        checkpoints = self.list_checkpoints(run_name)
        return checkpoints[-1] if checkpoints else None

    def ensure_cache_dirs(self) -> None:
        """Create the cache and checkpoint root directories if missing."""
        (self._cache_root / _HUB_SUBDIR).mkdir(parents=True, exist_ok=True)
        (self._cache_root / _DATASETS_SUBDIR).mkdir(parents=True, exist_ok=True)
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_dataset(resource: str) -> bool:
        """Heuristic: return True if *resource* looks like a dataset name."""
        lower = resource.lower()
        return "wiki" in lower or "dataset" in lower
