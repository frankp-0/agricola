# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Progress reporting for interactive and batch pipeline runs."""

from __future__ import annotations

import logging
import time

from tqdm import tqdm


class ProgressReporter:
    """Display terminal progress and log progress at fixed percentage milestones."""

    def __init__(
        self,
        total: int,
        unit: str,
        logger: logging.Logger,
        desc: str | None = None,
        milestone_percent: int = 5,
    ) -> None:
        self.total = total
        self.unit = unit
        self.logger = logger
        self.desc = desc
        self.milestone_percent = milestone_percent
        self.next_milestone = milestone_percent
        self.pbar = tqdm(total=total, desc=desc, unit=unit, disable=None)
        self.start_time = time.monotonic()
        self.logger.info("Starting %s.", self._label(0))

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.close(completed=exc_type is None)

    def update(self, n: int = 1) -> None:
        """Advance progress and log when crossing the next percentage milestone."""
        self.pbar.update(n)
        if self.total == 0 or self.pbar.n >= self.total:
            return

        percent = 100 * self.pbar.n / self.total
        if percent >= self.next_milestone:
            elapsed = time.monotonic() - self.start_time
            rate = self.pbar.n / elapsed if elapsed else 0.0
            remaining = elapsed * (self.total - self.pbar.n) / self.pbar.n
            self.logger.info(
                "Processed %s (%.1f%%, %.2f %s/s, ETA %s).",
                self._label(self.pbar.n),
                percent,
                rate,
                self.unit,
                self._format_duration(remaining),
            )
            self.next_milestone = (int(percent) // self.milestone_percent + 1) * (
                self.milestone_percent
            )

    def close(self, completed: bool = True) -> None:
        """Close the terminal display and record the final operation status."""
        self.pbar.close()
        elapsed = time.monotonic() - self.start_time
        status = "Completed" if completed else "Stopped"
        self.logger.info(
            "%s %s in %s.", status, self._label(self.pbar.n), self._format_duration(elapsed)
        )

    def _label(self, completed: int) -> str:
        prefix = f"{self.desc}: " if self.desc else ""
        return f"{prefix}{completed}/{self.total} {self.unit}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        rounded_seconds = round(seconds)
        minutes, seconds = divmod(rounded_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
