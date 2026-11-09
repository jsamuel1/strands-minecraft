"""NoteQueue — notes.ts (#32, #43)."""

from __future__ import annotations

from mcbrain.notes import NoteQueue, classify_note, note_source


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def q(**kw: int) -> tuple[NoteQueue, Clock]:
    c = Clock()
    return NoteQueue(clock=c, **{"cap": 40, "fresh_ms": 30_000, "usable_ms": 120_000, **kw}), c


def test_sightings_collapse_by_subject_with_a_count() -> None:
    n, c = q()
    for d in (6.2, 5.0, 4.1):
        n.push(f"(sentinel) A phantom closed to {d} blocks (inside 8)")
        c.t += 2_000
    out = n.take()
    assert out.count("\n") == 0
    assert "4.1 blocks" in out and "latest of 3 reports" in out


def test_durable_notes_are_stamped_not_dropped() -> None:
    n, c = q()
    n.push("(system) You DIED and respawned.")
    c.t += 10 * 60_000
    a = n.take_audited()
    assert "You DIED" in a.text and "(noted 10m ago)" in a.text and a.stamped == 1


def test_perishable_notes_rot_and_are_counted_by_source() -> None:
    n, c = q()
    n.push("(sentinel) A zombie closed to 2.0 blocks")
    c.t += 121_000
    a = n.take_audited()
    assert a.delivered == 0 and a.perished == 1 and a.sources == {"sentinel": 1}


def test_hedge_between_fresh_and_usable() -> None:
    n, c = q()
    n.push("(sentinel) A zombie closed to 2.0 blocks")
    c.t += 45_000
    assert "may not hold now" in n.take()


def test_cap_evicts_perishable_before_durable() -> None:
    n, _ = q(cap=2)
    n.push("(system) worker Digger failed")
    n.push("(sentinel) A zombie closed to 3 blocks")
    n.push("(system) reconnected")
    out = n.take()
    assert "zombie" not in out and "Digger" in out and "reconnected" in out
    assert n.evicted == 1


def test_wordless_notes_never_enter_the_rail() -> None:
    n, _ = q()
    assert n.push("   ") is False
    assert not n.has_pending()


def test_classify_and_source() -> None:
    assert classify_note("(sentinel) NIGHT falls")[0] == "perishable"
    assert classify_note("(system) You DIED")[0] == "durable"
    assert note_source("(reflex flee) ran") == "reflex"
