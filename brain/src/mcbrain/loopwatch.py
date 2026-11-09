"""Event-loop lag, watched. Port of ``src/loopwatch.ts``.

The keep-alive now lives in the Node sidecar, so a stalled Python loop cannot get the bot
kicked — but it still delays every note, reflex ask and journey step. A sampler cannot fix
starvation; it makes it attributable. Edge-triggered per episode.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

RISK_MS = 5_000
NOTICE_MS = 1_000


@dataclass(frozen=True)
class LagVerdict:
    report: bool
    level: str
    text: str


def classify_lag(lag_ms: float, culprit: str | None = None) -> LagVerdict:
    where = f" (while {culprit})" if culprit else ""
    s = f"{lag_ms / 1000:.1f}"
    if lag_ms >= RISK_MS:
        return LagVerdict(
            True,
            "risk",
            f"brain event loop stalled {s}s{where} — notes, reflex asks and journey steps all wait on this loop",
        )
    if lag_ms >= NOTICE_MS:
        return LagVerdict(True, "notice", f"brain event loop lag {s}s{where}")
    return LagVerdict(False, "quiet", "")


async def watch_loop(emit: Callable[[LagVerdict], None], sample_ms: int = 500, culprit: Callable[[], str | None] | None = None) -> None:
    armed = True
    while True:
        expected = time.monotonic() + sample_ms / 1000
        await asyncio.sleep(sample_ms / 1000)
        lag = (time.monotonic() - expected) * 1000
        v = classify_lag(lag, culprit() if culprit else None)
        if not v.report:
            armed = True
        elif armed:
            armed = False
            emit(v)
