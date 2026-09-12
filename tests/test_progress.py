# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import logging

from agricola.pipeline import progress


class FakeProgressBar:
    def __init__(self, **kwargs):
        self.n = 0
        self.kwargs = kwargs
        self.closed = False

    def update(self, n):
        self.n += n

    def close(self):
        self.closed = True


def test_progress_reporter_logs_five_percent_milestones(monkeypatch, caplog):
    times = iter([0.0, 10.0, 20.0, 30.0, 40.0])
    bars = []

    def make_progress_bar(**kwargs):
        bar = FakeProgressBar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(progress, "tqdm", make_progress_bar)
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(times))

    logger = logging.getLogger("test.progress")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with progress.ProgressReporter(20, "block", logger) as reporter:
            reporter.update()
            reporter.update()
            reporter.update(18)

    assert bars[0].kwargs["disable"] is None
    assert bars[0].closed
    assert [record.message for record in caplog.records] == [
        "Starting 0/20 block.",
        "Processed 1/20 block (5.0%, 0.10 block/s, ETA 3m 10s).",
        "Processed 2/20 block (10.0%, 0.10 block/s, ETA 3m 0s).",
        "Completed 20/20 block in 30s.",
    ]


def test_progress_reporter_logs_stopped_status_after_error(monkeypatch, caplog):
    times = iter([0.0, 10.0])

    monkeypatch.setattr(progress, "tqdm", lambda **kwargs: FakeProgressBar(**kwargs))
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(times))

    logger = logging.getLogger("test.progress")
    with caplog.at_level(logging.INFO, logger=logger.name):
        try:
            with progress.ProgressReporter(20, "block", logger):
                raise RuntimeError("interrupted")
        except RuntimeError:
            pass

    assert caplog.records[-1].message == "Stopped 0/20 block in 10s."
