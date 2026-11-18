import logging
from collections import Counter, deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from time import monotonic

import anyio
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)
model_deadline: ContextVar[float | None] = ContextVar("model_deadline", default=None)


class QueryBusy(RuntimeError):
    pass


class QueryAdmission:
    """Per-process admission for requests sharing the local query workers."""

    def __init__(self, total: int, per_scope: int):
        self.limit = total
        self.per_scope = per_scope
        self.active = 0
        self.rejected = 0
        self.scopes: Counter[str] = Counter()

    @contextmanager
    def reserve(self, scope: str):
        if self.active >= self.limit or self.scopes[scope] >= self.per_scope:
            self.rejected += 1
            raise QueryBusy
        self.active += 1
        self.scopes[scope] += 1
        try:
            yield
        finally:
            self.active -= 1
            self.scopes[scope] -= 1
            if not self.scopes[scope]:
                del self.scopes[scope]

    def status(self) -> dict:
        return {"in_flight": self.active, "limit": self.limit, "rejected": self.rejected}


class ModelUnavailable(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ModelGuard:
    """Bound upstream calls and permit one recovery probe after a breaker cooldown."""

    def __init__(self):
        self.slots = anyio.CapacityLimiter(settings.llm_max_inflight)
        self.queue_seconds = settings.llm_queue_seconds
        self.requests_per_minute = settings.llm_requests_per_minute
        self.recent_calls: deque[float] = deque()
        self.threshold = settings.llm_breaker_failures
        self.cooldown = settings.llm_breaker_cooldown
        self.open_until = 0.0
        self.probe_active = False
        self.generation = 0
        self.consecutive_failures = 0
        self.calls = 0
        self.failures = 0
        self.rejected = 0

    def status(self) -> dict:
        state = "closed"
        if self.open_until:
            state = "open" if monotonic() < self.open_until else "half_open"
        return {"state": state, "in_flight": self.slots.borrowed_tokens, "limit": self.slots.total_tokens,
                "requests_per_minute": self.requests_per_minute or None,
                "calls": self.calls, "failures": self.failures, "rejected": self.rejected,
                "probe_active": self.probe_active}

    def reject(self, reason: str):
        self.rejected += 1
        raise ModelUnavailable(reason)

    def disconnect(self, cooldown: float):
        self.open_until = monotonic() + cooldown
        self.generation += 1
        logger.warning("Model circuit opened for %.2fs", cooldown)

    def close_circuit(self):
        if self.open_until:
            logger.info("Model circuit closed after successful probe")
        self.open_until = 0.0
        self.consecutive_failures = 0

    def record_failure(self, exc: Exception, probe: bool, current: bool):
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        upstream_failure = status is None or status in {401, 403, 429} or status >= 500
        if not upstream_failure:
            # A rejected payload shows the service is reachable.
            if current:
                self.close_circuit()
            return
        self.failures += 1
        if not current:
            return
        self.consecutive_failures += 1
        if probe or status in {401, 403, 429} or self.consecutive_failures >= self.threshold:
            cooldown = self.cooldown
            if status == 429:
                retry_after = exc.response.headers.get("Retry-After", "")
                if retry_after.isdecimal() and len(retry_after) <= 8:
                    cooldown = max(cooldown, int(retry_after))
            self.disconnect(cooldown)

    @asynccontextmanager
    async def call(self, deadline: float):
        remaining = deadline - monotonic()
        if remaining <= 0:
            self.reject("model_budget")
        if self.open_until > monotonic() or self.probe_active:
            self.reject("model_circuit")
        try:
            self.slots.acquire_nowait()
        except anyio.WouldBlock:
            if not self.queue_seconds:
                self.reject("model_capacity")
            try:
                with anyio.fail_after(min(self.queue_seconds, remaining)):
                    await self.slots.acquire()
            except TimeoutError:
                self.reject("model_capacity")
        probe = False
        try:
            if self.open_until > monotonic() or self.probe_active:
                self.reject("model_circuit")
            remaining = deadline - monotonic()
            if remaining <= 0:
                self.reject("model_budget")
            if self.requests_per_minute:
                now = monotonic()
                while self.recent_calls and self.recent_calls[0] <= now - 60:
                    self.recent_calls.popleft()
                if len(self.recent_calls) >= self.requests_per_minute:
                    self.reject("model_rate")
                self.recent_calls.append(now)
            probe = bool(self.open_until)
            self.probe_active |= probe
            generation = self.generation
            self.calls += 1
            try:
                with anyio.fail_after(remaining):
                    yield
            except (httpx.HTTPError, TimeoutError) as exc:
                self.record_failure(exc, probe, generation == self.generation)
                if isinstance(exc, TimeoutError):
                    raise ModelUnavailable("model_budget") from exc
                raise
            else:
                if generation == self.generation:
                    self.close_circuit()
        finally:
            if probe:
                self.probe_active = False
            self.slots.release()
