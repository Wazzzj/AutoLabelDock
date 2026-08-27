"""Small, UI-independent FIFO queue for serial training batches."""
from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar


T = TypeVar("T")


class TrainingQueue(Generic[T]):
    """Track one active job and a FIFO list of waiting jobs.

    ``batch_had_multiple`` remains true for the lifetime of a batch once two
    jobs have coexisted.  The caller uses that stable fact to decide whether
    finished models may be auto-loaded.
    """

    def __init__(self) -> None:
        self._active: T | None = None
        self._waiting: deque[T] = deque()
        self._batch_had_multiple = False

    @property
    def active(self) -> T | None:
        return self._active

    @property
    def waiting(self) -> tuple[T, ...]:
        return tuple(self._waiting)

    @property
    def batch_had_multiple(self) -> bool:
        return self._batch_had_multiple

    def __len__(self) -> int:
        return (1 if self._active is not None else 0) + len(self._waiting)

    def enqueue(self, job: T) -> int:
        """Append ``job`` and return its one-based waiting position."""
        self._waiting.append(job)
        if len(self) > 1:
            self._batch_had_multiple = True
        return len(self._waiting)

    def start_next(self) -> T | None:
        """Move the first waiting job into the active slot."""
        if self._active is not None:
            raise RuntimeError("a training job is already active")
        if not self._waiting:
            return None
        self._active = self._waiting.popleft()
        return self._active

    def finish_active(self) -> tuple[T | None, bool]:
        """Release the active job and return it plus its batch policy."""
        job = self._active
        had_multiple = self._batch_had_multiple
        self._active = None
        if not self._waiting:
            self._batch_had_multiple = False
        return job, had_multiple

    def clear_waiting(self) -> list[T]:
        """Remove and return all jobs that have not started yet."""
        jobs = list(self._waiting)
        self._waiting.clear()
        if self._active is None:
            self._batch_had_multiple = False
        return jobs
