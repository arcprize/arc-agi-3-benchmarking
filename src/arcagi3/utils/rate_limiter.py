import asyncio
import time


class AsyncRequestRateLimiter:
    """Asynchronous request-based rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, capacity: float):
        if not isinstance(rate, (int, float)) or not rate > 0:
            raise ValueError("Rate must be a positive number")
        if not isinstance(capacity, (int, float)) or not capacity >= 0:
            raise ValueError("Capacity must be a non-negative number")
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._available_requests = self._capacity
        self._last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        if elapsed > 0:
            new_requests_allowance = elapsed * self._rate
            self._available_requests = min(
                self._capacity, self._available_requests + new_requests_allowance
            )
            self._last_refill_time = now

    async def acquire(self, requests_needed: int = 1) -> None:
        if not isinstance(requests_needed, int) or requests_needed <= 0:
            raise ValueError("requests_needed must be a positive integer")
        if requests_needed > self._capacity:
            raise ValueError(
                f"Requested requests ({requests_needed}) exceeds bucket capacity ({self._capacity})"
            )

        while True:
            async with self._lock:
                self._refill()
                if self._available_requests >= requests_needed:
                    self._available_requests -= requests_needed
                    return
                needed = requests_needed - self._available_requests
                wait_time = (needed / self._rate) if self._rate > 0 else 3600.0
            await asyncio.sleep(wait_time)

    async def __aenter__(self):
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_available_requests(self) -> float:
        async with self._lock:
            self._refill()
            return self._available_requests
