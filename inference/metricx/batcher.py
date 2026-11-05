"""A vLLM-inspired dynamic request batcher.

The HTTP layer accepts unbounded concurrent requests and drops each scoring
item onto a single queue; one background worker drains the queue into a
cross-request batch (bounded by ``max_batch`` items or a ``max_wait_ms``
window, whichever comes first) and runs one blocking inference call over the
whole batch. That funnel is exactly vLLM's "one scheduler, one model, dynamic
batches" idea, minus paged attention (MetricX runs a single decoder step, not
autoregressive generation), and it is what turns many small concurrent
requests into GPU-efficient batches without ever running two forward passes at
once.

The module imports nothing from torch or DataCurator: ``infer_fn`` is an opaque
blocking callable, so the batcher can be unit-tested with a plain Python stub.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

# infer_fn maps a batch of opaque items to one score per item, positionally.
# It is *blocking* (a torch forward pass) and is run off the event loop.
InferFn = Callable[[List[T]], Sequence[float]]


class DynamicBatcher:
    """Coalesces concurrently-submitted items into batched inference calls."""

    def __init__(
        self,
        infer_fn: InferFn,
        *,
        max_batch: int = 64,
        max_wait_ms: float = 8.0,
        on_batch: Optional[Callable[[int], None]] = None,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._infer_fn = infer_fn
        self._max_batch = max_batch
        self._max_wait = max_wait_ms / 1000.0
        self._on_batch = on_batch
        self._q: "asyncio.Queue[Tuple[T, asyncio.Future]]" = asyncio.Queue()

    def pending(self) -> int:
        """Current queue depth (for the queue-depth gauge)."""
        return self._q.qsize()

    async def submit(self, item: T) -> float:
        """Enqueue one item and await its score.

        Callers with several items (a request's ``pairs``) submit each
        separately and ``asyncio.gather`` the results, so they merge into the
        same cross-request batches as everyone else.
        """
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._q.put((item, fut))
        return await fut

    async def run(self) -> None:
        """The single inference worker. Start once as a background task."""
        while True:
            item, fut = await self._q.get()
            batch: List[Tuple[T, asyncio.Future]] = [(item, fut)]
            deadline = time.monotonic() + self._max_wait
            while len(batch) < self._max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._q.get(), timeout))
                except asyncio.TimeoutError:
                    break

            if self._on_batch is not None:
                self._on_batch(len(batch))

            items = [it for it, _ in batch]
            try:
                # Offload the blocking forward pass so the event loop keeps
                # accepting requests and forming the next batch while the GPU
                # is busy (request/inference pipelining).
                scores = await asyncio.to_thread(self._infer_fn, items)
            except Exception as exc:  # noqa: BLE001 — propagate to every waiter
                for _, f in batch:
                    if not f.done():
                        f.set_exception(exc)
                continue

            for (_, f), score in zip(batch, scores):
                if not f.done():
                    f.set_result(float(score))
