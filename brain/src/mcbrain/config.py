"""Every tunable in one registry — the Python twin of ``src/config.ts`` (see HARDCODING.md).

Class-(b) knobs only: budgets, windows, cadences a deployment may want to turn. Each
has an env override and a one-line reason for its default. Protocol constants stay next
to the code that speaks the protocol; per-situation decisions belong to the model.

Values are read at import, like the TS registry. Tests that need different numbers
construct the objects that take them (``NoteQueue(cap=…)``) instead of re-importing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


def num(name: str, fallback: float) -> float:
    """Env number with fallback; empty or garbage falls back rather than NaN-ing a timer."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if math.isfinite(value) else fallback


def integer(name: str, fallback: int) -> int:
    return int(num(name, fallback))


def string(name: str, fallback: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else fallback


def flag(name: str, fallback: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def names(name: str, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or fallback


@dataclass(frozen=True)
class SessionCfg:
    #: SESSION_WINDOW — 120 messages; floor 8, below which one agentic exchange cannot fit.
    window: int = field(default_factory=lambda: max(8, integer("SESSION_WINDOW", 120)))
    #: THROTTLE_RETRY_MS — one jittered retry after a throttle on a shared quota, then out.
    throttle_retry_ms: int = field(default_factory=lambda: max(0, integer("THROTTLE_RETRY_MS", 2_000)))
    #: HISTORY_AUDIT — walk the whole history after every fold/trim (issue #14 hunting).
    audit: bool = field(default_factory=lambda: flag("HISTORY_AUDIT"))


@dataclass(frozen=True)
class NotesCfg:
    #: NOTES_CAP — 40 pending notes; beyond it sightings are evicted before consequences.
    cap: int = field(default_factory=lambda: integer("NOTES_CAP", 40))
    #: NOTES_FRESH_MS — 30s verbatim; older notes are stamped with their age.
    fresh_ms: int = field(default_factory=lambda: integer("NOTES_FRESH_MS", 30_000))
    #: NOTES_USABLE_MS — 120s: a perishable note past this never reaches the mind.
    usable_ms: int = field(default_factory=lambda: integer("NOTES_USABLE_MS", 120_000))


@dataclass(frozen=True)
class JourneyCfg:
    #: JOURNEY_MAX_ITERATIONS — runaway guard; hit only when lost.
    max_iterations: int = field(default_factory=lambda: integer("JOURNEY_MAX_ITERATIONS", 500))
    #: JOURNEY_MAX_WALL_MS — 2h: longer than any real errand, short enough to notice a loop.
    max_wall_ms: int = field(default_factory=lambda: integer("JOURNEY_MAX_WALL_MS", 2 * 60 * 60 * 1000))
    #: JOURNEY_COOLDOWN_MS — 3s between steps: a journey must not monopolise the model.
    cooldown_ms: int = field(default_factory=lambda: integer("JOURNEY_COOLDOWN_MS", 3_000))
    #: JOURNEY_YIELD_MAX_MS — 60s max a step defers to live turns before proceeding anyway.
    yield_max_ms: int = field(default_factory=lambda: integer("JOURNEY_YIELD_MAX_MS", 60_000))
    #: JOURNEY_MAX_WAITS — 6 consecutive [WAITING:] steps before patience reads as a stall.
    max_consecutive_waits: int = field(default_factory=lambda: integer("JOURNEY_MAX_WAITS", 6))
    #: JOURNEY_WAIT_COOLDOWN_MS — 15s: a waiting step should not buy a model call in 3s.
    wait_cooldown_ms: int = field(default_factory=lambda: integer("JOURNEY_WAIT_COOLDOWN_MS", 15_000))
    #: JOURNEY_KEEP — newest journeys worth keeping on disk.
    keep: int = field(default_factory=lambda: integer("JOURNEY_KEEP", 10))


@dataclass(frozen=True)
class ThinkerCfg:
    #: THINKER_INTERVAL_MS — 90s idle-reflection cadence.
    interval_ms: int = field(default_factory=lambda: integer("THINKER_INTERVAL_MS", 90_000))
    #: THINKER_DISABLED — calm the idle mind (shared quota, benchmarks).
    disabled: bool = field(default_factory=lambda: flag("THINKER_DISABLED"))


@dataclass(frozen=True)
class ReflexCfg:
    #: REFLEX_COOLDOWN_MS — 15s: a burst of danger buys one focused turn, not a fork storm.
    cooldown_ms: int = field(default_factory=lambda: integer("REFLEX_COOLDOWN_MS", 15_000))
    #: REFLEX_DISABLED — no urgent turns from the body (the body's own spinal cord still runs).
    disabled: bool = field(default_factory=lambda: flag("REFLEX_DISABLED"))


@dataclass(frozen=True)
class SidecarCfg:
    #: SIDECAR_URL — the Node body sidecar (src/sidecar.ts); loopback by default.
    url: str = field(default_factory=lambda: string("SIDECAR_URL", "http://127.0.0.1:3010"))
    #: SIDECAR_POLL_MS — 20s long-poll: under the sidecar's 25s ceiling.
    poll_ms: int = field(default_factory=lambda: integer("SIDECAR_POLL_MS", 20_000))
    #: SIDECAR_TOOL_TIMEOUT_S — 180s: dig_vein/attack defaults run up to 90s, plus headroom.
    tool_timeout_s: int = field(default_factory=lambda: integer("SIDECAR_TOOL_TIMEOUT_S", 180))


@dataclass(frozen=True)
class PolicyCfg:
    #: POLICY_CANDIDATES — 12 macro-actions per decision: enough to cover the tech step
    #: in reach, few enough that a Choice's confidence still means something.
    max_candidates: int = field(default_factory=lambda: integer("POLICY_CANDIDATES", 12))
    #: POLICY_STEP_COOLDOWN_MS — 250ms between decisions: the body needs a tick to settle.
    step_cooldown_ms: int = field(default_factory=lambda: integer("POLICY_STEP_COOLDOWN_MS", 250))
    #: POLICY_SEARCH_RADIUS — 48 blocks: past the loaded view at VIEW_DISTANCE 6 finds nothing.
    search_radius: int = field(default_factory=lambda: integer("POLICY_SEARCH_RADIUS", 48))
    #: POLICY_STATE_CHARS — 6000 chars ≈ 1.5k tokens: the default state budget, which fits CLM's
    #: 2,048-token row with its question. Larger-window models get their own budget (below).
    state_chars: int = field(default_factory=lambda: integer("POLICY_STATE_CHARS", 6_000))
    #: POLICY_STATE_CHARS_KEV — 14000 chars: Kev's row is 8,192 tokens for state plus its longest
    #: question. MEASURED with Kev's own tokenizer (Qwen3.5-4B, 370 real v3 states): 2.11–2.74 chars
    #: per token, not 4. At the worst 2.11, 14000 chars is ~6.6k tokens, leaving ~1.5k for the question
    #: (observed 240). v3 run 1 used 24000 (~11k tokens) and got 1,087 HTTP 422 "row limit 8192"
    #: rejections on night, where the state grew past ~19.3k chars.
    state_chars_kev: int = field(default_factory=lambda: integer("POLICY_STATE_CHARS_KEV", 14_000))
    #: POLICY_STATE_CHARS_HOBSON — 7500 chars: hobson-2b-v19 serves a 4,096-token window (trained at
    #: 3,072) and SILENTLY truncates a longer state, so the budget must fit, not overflow. Its torso is
    #: Qwen3.5 (the tokenizer family measured at 2.11–2.74 chars/token on v3 states): 7,500 chars is at
    #: most ~3.6k tokens, leaving ~500 for the 12-option question. Every call records usage.input_tokens.
    state_chars_hobson: int = field(default_factory=lambda: integer("POLICY_STATE_CHARS_HOBSON", 7_500))
    #: POLICY_HISTORY_STEPS_HOBSON — 12 aligned steps before the 7.5k-char budget trims.
    history_steps_hobson: int = field(default_factory=lambda: integer("POLICY_HISTORY_STEPS_HOBSON", 12))
    #: POLICY_STATE_CHARS_JEV — 48000 chars ≈ 17–23k tokens at the measured 2.1–2.7 chars/token:
    #: under Jev's 32k-token row (v3 run 1 max: 45.6k chars ≈ 21.6k tokens, no rejections).
    state_chars_jev: int = field(default_factory=lambda: integer("POLICY_STATE_CHARS_JEV", 48_000))
    #: POLICY_HISTORY_STEPS — 12 aligned steps (recent_actions + previous_states) for CLM, before the
    #: state budget trims further. v1/v2 showed 4; Jev's 130× shelter loop and CLM's tree/wait
    #: alternation both ran past 4.
    history_steps: int = field(default_factory=lambda: integer("POLICY_HISTORY_STEPS", 12))
    #: POLICY_HISTORY_STEPS_CLM — 6 steps for CLM: its encoder reads 2,048 tokens, and the v3 smoke
    #: (2026-09-28) showed 12 steps + previous_states tripled server latency (~85 → ~600 ms) while it
    #: waited under attack. Six recent lines plus the untrimmed action_tally keep the facts.
    history_steps_clm: int = field(default_factory=lambda: integer("POLICY_HISTORY_STEPS_CLM", 6))
    #: POLICY_EVENT_CHARS_CLM — 60 chars per event for CLM: "hurt: … TOOK DAMAGE from ZOMBIE — health 9/20".
    event_chars_clm: int = field(default_factory=lambda: integer("POLICY_EVENT_CHARS_CLM", 60))
    #: POLICY_OUTCOME_CHARS_CLM — 120 chars per outcome for CLM (the change/verdict/reason carry the facts).
    outcome_chars_clm: int = field(default_factory=lambda: integer("POLICY_OUTCOME_CHARS_CLM", 120))
    #: POLICY_HISTORY_STEPS_KEV — 32 steps before the 14k-char budget trims (it usually binds first).
    history_steps_kev: int = field(default_factory=lambda: integer("POLICY_HISTORY_STEPS_KEV", 32))
    #: POLICY_HISTORY_STEPS_JEV — 64 steps: longer than any non-night episode, so Jev and LLMs on the
    #: same alphabet see (nearly) the whole episode.
    history_steps_jev: int = field(default_factory=lambda: integer("POLICY_HISTORY_STEPS_JEV", 64))
    #: POLICY_OUTCOME_CHARS — 240 chars of each step's outcome, after boilerplate is stripped.
    #: v1/v2 kept 140 raw, and "Equipped … confirmed in the slot." alone used 55 of them.
    outcome_chars: int = field(default_factory=lambda: integer("POLICY_OUTCOME_CHARS", 240))
    #: POLICY_MAX_EVENTS — 6 body events per step (after ×N dedup): radar/drowning/reflex notes
    #: arrive in bursts (v2 logged 32 drowning lines across 45 episodes) and must not crowd out history.
    max_events: int = field(default_factory=lambda: integer("POLICY_MAX_EVENTS", 6))
    #: POLICY_EVENT_CHARS — 120 chars per event line: enough for "radar: zombie inside 8 blocks — 5 away".
    event_chars: int = field(default_factory=lambda: integer("POLICY_EVENT_CHARS", 120))
    #: POLICY_EXPLORE_MIN_MOVE — 8 blocks moved counts as an exploration step done (the target is 24).
    explore_min_move: float = field(default_factory=lambda: num("POLICY_EXPLORE_MIN_MOVE", 8.0))
    #: POLICY_WAIT_S — 5s of game time per `wait` (≈100 ticks): a night is ~7 min, so a waiting
    #: arm spends ~80 decisions on it at 5s — enough to get to dawn inside the night task's budget.
    wait_s: float = field(default_factory=lambda: num("POLICY_WAIT_S", 5.0))
    #: POLICY_INSTRUCTIONS — short | long: short is equal-or-best for Jev, Kev and CLM in the
    #: 2026-09-26 ablation (policy.py INSTRUCTION_VARIANTS); every arm gets the same one.
    instructions: str = field(default_factory=lambda: string("POLICY_INSTRUCTIONS", "short"))


@dataclass(frozen=True)
class Config:
    session: SessionCfg = field(default_factory=SessionCfg)
    notes: NotesCfg = field(default_factory=NotesCfg)
    journey: JourneyCfg = field(default_factory=JourneyCfg)
    thinker: ThinkerCfg = field(default_factory=ThinkerCfg)
    reflex: ReflexCfg = field(default_factory=ReflexCfg)
    sidecar: SidecarCfg = field(default_factory=SidecarCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    #: PEER_BOTS — crew bots whose chat is logged, never answered.
    peer_bots: tuple[str, ...] = field(default_factory=lambda: names("PEER_BOTS"))


cfg = Config()
