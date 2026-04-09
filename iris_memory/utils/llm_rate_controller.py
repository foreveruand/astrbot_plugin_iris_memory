"""
LLM Rate Controller - Controls concurrent LLM calls to avoid API rate limits

Provides two-layer protection:
1. Semaphore: limits max concurrent calls
2. Minimum interval: enforces time gap between calls to same provider
"""

from __future__ import annotations

import asyncio
import time

from iris_memory.utils.logger import get_logger

logger = get_logger("llm_rate_controller")


class LLMRateController:
    """Global rate controller for all LLM calls in the plugin.

    Singleton pattern ensures single point of control across all modules.

    Usage:
        controller = LLMRateController.get_instance()
        async with controller.acquire("google_gemini/gemini-3-flash"):
            result = await llm_call(...)
    """

    _instance: LLMRateController | None = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> LLMRateController:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_concurrent: int = 3,
        min_interval_ms: int = 1000,
    ) -> None:
        """Initialize the rate controller.

        Args:
            max_concurrent: Maximum concurrent LLM calls across all providers
            min_interval_ms: Minimum interval between calls to the same provider (ms)
        """
        if self._initialized:
            return

        self._max_concurrent = max_concurrent
        self._min_interval = min_interval_ms / 1000.0
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_call_time: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._initialized = True

        logger.info(
            f"LLMRateController initialized: max_concurrent={max_concurrent}, "
            f"min_interval={min_interval_ms}ms"
        )

    @classmethod
    def get_instance(cls) -> LLMRateController:
        """Get the singleton instance, creating if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def configure(cls, max_concurrent: int, min_interval_ms: int) -> None:
        """Reconfigure the rate controller.

        This recreates the singleton with new settings.
        """
        cls._instance = None
        cls._initialized = False
        cls(max_concurrent=max_concurrent, min_interval_ms=min_interval_ms)

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def min_interval_ms(self) -> int:
        return int(self._min_interval * 1000)

    def _get_provider_key(self, provider_id: str | None) -> str:
        """Normalize provider ID for tracking."""
        if not provider_id:
            return "default"
        return provider_id

    async def acquire(self, provider_id: str | None = None) -> None:
        """Acquire rate limit slot, may wait for both semaphore and interval.

        Args:
            provider_id: LLM provider identifier for per-provider rate limiting
        """
        provider_key = self._get_provider_key(provider_id)

        async with self._lock:
            last_time = self._last_call_time.get(provider_key, 0.0)
            elapsed = time.time() - last_time
            wait_time = self._min_interval - elapsed

            if wait_time > 0:
                logger.debug(
                    f"Rate control: waiting {wait_time:.3f}s for provider {provider_key}"
                )
                await asyncio.sleep(wait_time)

        await self._semaphore.acquire()
        logger.debug(f"Rate control: acquired slot for {provider_key}")

    def release(self, provider_id: str | None = None) -> None:
        """Release rate limit slot and record call time.

        Args:
            provider_id: LLM provider identifier
        """
        provider_key = self._get_provider_key(provider_id)
        self._last_call_time[provider_key] = time.time()
        self._semaphore.release()
        logger.debug(f"Rate control: released slot for {provider_key}")

    async def __aenter__(self) -> LLMRateController:
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        self.release()

    def get_stats(self) -> dict[str, object]:
        """Get current rate controller statistics."""
        return {
            "max_concurrent": self._max_concurrent,
            "min_interval_ms": self.min_interval_ms,
            "semaphore_value": self._semaphore._value,
            "tracked_providers": len(self._last_call_time),
        }


class RateControlledContext:
    """Context manager for rate-controlled LLM calls.

    Usage:
        async with RateControlledContext(provider_id) as ctx:
            result = await llm_call(...)
    """

    def __init__(self, provider_id: str | None = None) -> None:
        self._controller = LLMRateController.get_instance()
        self._provider_id = provider_id

    async def __aenter__(self) -> RateControlledContext:
        await self._controller.acquire(self._provider_id)
        return self

    async def __aexit__(self, *args) -> None:
        self._controller.release(self._provider_id)


def get_rate_controller() -> LLMRateController:
    """Convenience function to get the singleton rate controller."""
    return LLMRateController.get_instance()
