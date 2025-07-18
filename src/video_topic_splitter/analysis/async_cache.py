"""Async-safe cache implementation for topic analysis."""

import asyncio
from collections import deque
from typing import Dict, Optional, Any

from .topic_analyzer_config import DEFAULT_CACHE_SIZE


class AsyncCache:
    """Async-safe cache implementation for topic analysis results."""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: deque = deque()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value asynchronously."""
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set cached value asynchronously."""
        async with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key] = value
                self._access_order.remove(key)
                self._access_order.append(key)
            else:
                # Add new
                if len(self._cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = self._access_order.popleft()
                    del self._cache[oldest_key]
                
                self._cache[key] = value
                self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()