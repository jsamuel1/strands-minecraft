"""The note rail's queue — what the senses owe the MIND. Port of ``src/notes.ts`` (#32, #43).

Notes ride in FRONT of the next turn, never into a possibly mid-tool-call history. Capped,
aged and collapsed by subject: fourteen sightings of one phantom are one fact with a count.
Two windows per class — verbatim inside ``fresh_ms``; stamped with age after it; a perishable
sighting past ``usable_ms`` never reaches the mind and is counted as perished, by source.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .config import cfg

NoteClass = Literal["perishable", "durable"]


@dataclass
class NoteEntry:
    text: str
    cls: NoteClass
    subject: str | None
    first_at: float
    last_at: float
    count: int = 1


def note_source(text: str) -> str:
    """The sink that wrote a note, from its own prefix: ``(sentinel) …`` → ``sentinel``."""
    m = re.match(r"^\(([^)]{1,24})\)", text.strip())
    return m.group(1).strip().split()[0].lower() if m else "unknown"


def note_age(ms: float) -> str:
    s = round(ms / 1000)
    return f"{s}s" if s < 90 else f"{round(s / 60)}m"


_RADAR = re.compile(r"^\(sentinel\)\s+(?:A|An|The)\s+([a-z0-9_ ]+?)\s+(?:closed to|pulled back to|is\b)", re.I)
_SECURITY = re.compile(r"^\(([^)]*)\)\s*(\S+)\s+(?:just OPENED|is BREAKING|is STILL digging)")


def classify_note(text: str) -> tuple[NoteClass, str | None]:
    """Only sightings are perishable; when in doubt a note is durable (never lose a death notice)."""
    if m := _RADAR.match(text):
        return "perishable", f"mob:{m.group(1).strip().lower()}"
    if m := _SECURITY.match(text):
        return "durable", f"security:{m.group(2).lower()}"
    if re.match(r"^\(sentinel\)\s+(DAWN|NIGHT|DUSK)", text, re.I):
        return "perishable", "clock"
    if re.match(r"^\(reflex", text, re.I):
        return "perishable", "reflex"
    return "durable", None


def render_note(e: NoteEntry, now: float, fresh_ms: float = float("inf")) -> str:
    body = e.text
    if e.count > 1:
        span = round((e.last_at - e.first_at) / 1000)
        ago = round((now - e.last_at) / 1000)
        body = f"{e.text} [latest of {e.count} reports over {span}s, last {ago}s ago]"
    age = now - e.last_at
    if age < fresh_ms:
        return body
    if e.cls == "perishable":
        return f"{body} (seen {note_age(age)} ago — was true then, may not hold now)"
    return f"{body} (noted {note_age(age)} ago)"


@dataclass
class Audited:
    text: str
    delivered: int
    stamped: int
    perished: int
    sources: dict[str, int] = field(default_factory=dict)


def _ms() -> float:
    return time.time() * 1000


class NoteQueue:
    def __init__(
        self,
        cap: int = cfg.notes.cap,
        fresh_ms: int = cfg.notes.fresh_ms,
        clock: Callable[[], float] = _ms,
        usable_ms: int = cfg.notes.usable_ms,
    ) -> None:
        self.cap = cap
        self.fresh_ms = fresh_ms
        self.clock = clock
        # A usable window shorter than the fresh one means "stamp it, then drop it".
        self.usable_ms = max(usable_ms, fresh_ms)
        self._queue: list[NoteEntry] = []
        self.evicted = 0
        self.stale_dropped = 0
        self.collapsed = 0
        self.perished = 0
        self._perished_sources: dict[str, int] = {}

    def push(self, text: str, *, cls: NoteClass | None = None, subject: str | None = None) -> bool:
        """Queue a note. False when it was folded into one already waiting (or was wordless, #39)."""
        if not (text or "").strip():
            return False
        now = self.clock()
        c, s = classify_note(text)
        cls = cls or c
        subject = subject if subject is not None else s
        same = next((e for e in self._queue if (e.subject == subject if subject is not None else e.text == text)), None)
        if same:
            same.text = text  # the newest wording carries the current distance
            same.last_at = now
            same.count += 1
            self.collapsed += 1
            return False
        self._queue.append(NoteEntry(text, cls, subject, now, now))
        while len(self._queue) > self.cap:
            idx = next((i for i, e in enumerate(self._queue) if e.cls == "perishable"), 0)
            del self._queue[idx]
            self.evicted += 1
        return True

    def flush_stale(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        kept: list[NoteEntry] = []
        dropped = 0
        for e in self._queue:
            if e.cls == "durable" or now - e.last_at < self.usable_ms:
                kept.append(e)
                continue
            dropped += 1
            self.perished += 1
            src = note_source(e.text)
            self._perished_sources[src] = self._perished_sources.get(src, 0) + 1
        self._queue = kept
        self.stale_dropped += dropped
        return dropped

    def has_pending(self) -> bool:
        self.flush_stale()
        return bool(self._queue)

    def take(self) -> str:
        return self.take_audited().text

    def take_audited(self) -> Audited:
        now = self.clock()
        self.flush_stale(now)
        stamped = sum(1 for e in self._queue if now - e.last_at >= self.fresh_ms)
        out = [render_note(e, now, self.fresh_ms) for e in self._queue]
        self._queue = []
        sources = dict(self._perished_sources)
        self._perished_sources.clear()
        return Audited("\n".join(out), len(out), stamped, sum(sources.values()), sources)

    def stats(self) -> dict[str, object]:
        now = self.clock()
        self.flush_stale(now)
        return {
            "pending": len(self._queue),
            "perishable": sum(1 for e in self._queue if e.cls == "perishable"),
            "cap": self.cap,
            "evicted": self.evicted,
            "staleDropped": self.stale_dropped,
            "collapsed": self.collapsed,
            "oldestAgeMs": max((now - e.last_at for e in self._queue), default=None),
            "perished": self.perished,
            "freshMs": self.fresh_ms,
            "usableMs": self.usable_ms,
        }
