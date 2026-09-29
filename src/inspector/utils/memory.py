"""Peak host memory per pipeline stage (docs/13, F07).

On this project host RAM, not VRAM, is the binding constraint: the patch
matrix at native resolution is tens of gigabytes while the forward pass is
under 1 GB. A single end-of-run number cannot say which stage peaked, so each
stage is wrapped in a sampler that polls this process's resident set size.

Sampling can miss a spike shorter than its interval; the interval is recorded
with the result. Without `psutil` the peak is NaN rather than a guess.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from types import TracebackType


class PeakRSS:
    """Context manager: peak resident memory (MB) of this process while open."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_mb = math.nan
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_mb: Callable[[], float] | None = None

    def _sample(self, rss_mb: Callable[[], float]) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakRSS:
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()

        def rss_mb() -> float:
            return float(process.memory_info().rss) / 1024**2

        self._rss_mb = rss_mb
        self.peak_mb = rss_mb()
        self._thread = threading.Thread(target=self._sample, args=(rss_mb,), name="peak-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._thread is not None and self._rss_mb is not None:
            self._stop.set()
            self._thread.join()
            self.peak_mb = max(self.peak_mb, self._rss_mb())


class PeakVRAM:
    """Context manager: peak allocated CUDA memory (MB) while open; NaN without CUDA."""

    def __init__(self) -> None:
        self.peak_mb = math.nan
        self._cuda = False

    def __enter__(self) -> PeakVRAM:
        try:
            import torch
        except Exception:
            return self
        if torch.cuda.is_available():
            self._cuda = True
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._cuda:
            import torch

            torch.cuda.synchronize()
            self.peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
