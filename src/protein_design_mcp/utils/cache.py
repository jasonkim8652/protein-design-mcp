"""
Result caching utilities.

Caches computation results to avoid redundant expensive operations.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheConfig:
    """Configuration for result cache."""

    cache_dir: Path = Path(os.environ.get("CACHE_DIR", "~/.cache/protein-design-mcp"))
    max_size_gb: float = 10.0
    ttl_days: int = 30


class ResultCache:
    """Cache for computation results."""

    def __init__(self, config: CacheConfig | None = None):
        """Initialize cache."""
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, operation: str, params: dict[str, Any]) -> str:
        """Compute cache key from operation and parameters."""
        key_data = json.dumps({"operation": operation, "params": params}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get(self, operation: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get cached result.

        Args:
            operation: Operation name (e.g., "esmfold")
            params: Operation parameters

        Returns:
            Cached result or None if not found
        """
        key = self._compute_key(operation, params)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None

        return None

    def set(self, operation: str, params: dict[str, Any], result: dict[str, Any]) -> None:
        """
        Store result in cache.

        Args:
            operation: Operation name
            params: Operation parameters
            result: Result to cache
        """
        key = self._compute_key(operation, params)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except IOError:
            pass  # Fail silently on cache write errors

    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass

    def get_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                total += cache_file.stat().st_size
            except IOError:
                pass
        return total
