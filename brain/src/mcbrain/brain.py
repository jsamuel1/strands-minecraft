"""The LLM brain's spine — port of ``src/index.ts`` minus voice and the dashboard.

One shared ``Session`` over an ``Agent`` whose tools are the body's 54 (over MCP) plus the
journey tools. The sidecar's event stream feeds the rails:

- ``note`` / ``revive`` / ``death`` → the NoteQueue (rides in front of the next turn)
- ``reflex`` → an urgent turn through one gate: one in flight plus ``REFLEX_COOLDOWN_MS``
- ``chat`` → a turn (the sidecar already dropped peer bots and classified server lines)
- ``gave_up`` → corpse guard: stop every rail
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookProvider, HookRegistry
from strands.models import Model
from strands.types.content import Messages

from .config import cfg
from .journeys import BodySnapshot, JourneyRunner, journey_tools
from .notes import NoteQueue
from .session import Session
from .sidecar import Body, SidecarEvent
from .thinker import Thinker

logger = logging.getLogger(__name__)

TURN_ECONOMY = """One call beats ten — several tools finish a whole job; reach for them before looping:
- find_blocks groups veins/trees into clusters (veinSize, exposed) and takes comma
  alternatives ('coal_ore,deepslate_coal_ore', all log species in one search).
- dig_vein fells an entire tree or mines a whole vein from ONE starting block and sweeps
  the drops. Never issue dig_block per log/ore.
- craft_item chains intermediates by itself (logs → planks → sticks → pickaxe) and, when
  short, names exactly what to gather and how many. 3x3 recipes need a crafting table within
  4 blocks — place or walk to one first.
- attack_entity with until:'dead' fights the whole duel.
- build_structure drafts AND builds basic shells (box/wall/floor/pillar) from dimensions.
- remember_place/recall_places waypoints: save home, base, farms once."""

SYSTEM_PROMPT = f"""You are a Minecraft bot embodied in a live game world via mineflayer.
Your username in game is "{{username}}".

You have real tools: perception (look around, find blocks, inventory), movement
(pathfinding, turning, raw controls), digging, building, crafting, combat,
containers, sleeping, and chat.

Behavior rules:
- ACT, don't narrate. Call the tool.
- Perceive before acting when the request depends on world state you haven't verified.
- Chain tools to finish multi-step jobs.
- Report results concisely. Mention coordinates when useful.
- If something fails, read the error — it usually names what's missing — and retry sensibly.
- NEVER ask a player for their coordinates — use find_player.
- Never toss your tools or armor unless explicitly asked.

{TURN_ECONOMY}
- A long errand belongs in a journey: start_journey runs it step by step in the background."""


@dataclass
class ModelCallStats:
    """Per model-call latency and failures, for p50/p95 and the throttle column."""

    latencies_ms: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    _started: float | None = None

    def before(self, _event: BeforeModelCallEvent) -> None:
        self._started = time.perf_counter()

    def after(self, event: AfterModelCallEvent) -> None:
        if self._started is not None:
            self.latencies_ms.append((time.perf_counter() - self._started) * 1000)
            self._started = None
        if event.exception is not None:
            name = type(event.exception).__name__
            self.errors[name] = self.errors.get(name, 0) + 1


class _StatsHook(HookProvider):
    def __init__(self, stats: ModelCallStats) -> None:
        self.stats = stats

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        stats = self.stats

        def before(event: BeforeModelCallEvent) -> Awaitable[None] | None:
            stats.before(event)
            return None

        def after(event: AfterModelCallEvent) -> Awaitable[None] | None:
            stats.after(event)
            return None

        registry.add_callback(BeforeModelCallEvent, before)
        registry.add_callback(AfterModelCallEvent, after)


def build_agent(
    model: Model, body_tools: list[Any], username: str, extra_tools: list[Any], stats: ModelCallStats
) -> tuple[Agent, Callable[[Messages, NullConversationManager], Agent]]:
    """The root Agent and its fork factory: same model/prompt/tools, seeded history, no self-trim."""
    prompt = SYSTEM_PROMPT.replace("{username}", username)
    tools = [*body_tools, *extra_tools]
    hook = _StatsHook(stats)
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        callback_handler=None,
        hooks=[hook],
        conversation_manager=NullConversationManager(),
    )

    def fork(seed: Messages, manager: NullConversationManager) -> Agent:
        # Shared tool objects on purpose: every fork drives the same body.
        return Agent(
            model=model,
            tools=tools,
            system_prompt=prompt,
            messages=seed,
            callback_handler=None,
            hooks=[_StatsHook(stats)],
            conversation_manager=manager,
        )

    return agent, fork


class Brain:
    """The rails, wired. ``pump()`` drains the sidecar's events; ``run(text)`` is one turn."""

    def __init__(
        self,
        model: Model,
        body: Body,
        *,
        username: str | None = None,
        thinker: bool | None = None,
        body_tools: list[Any] | None = None,
    ) -> None:
        self.body = body
        self.stats = ModelCallStats()
        self.notes = NoteQueue()
        self.journeys = JourneyRunner()
        self.username = username or body.state().get("username") or body.name
        # The body's 54 tools arrive over MCP; tests pass their own.
        tools = body_tools if body_tools is not None else [body.agent_mcp_client()]
        agent, fork = build_agent(model, tools, self.username, journey_tools(self.journeys), self.stats)
        self.session = Session(agent, fork)
        self.session.on_diagnosis = lambda r: logger.warning("%s", r)
        self.journeys.bind(self.session)
        self.journeys.snapshot = self._snapshot
        self.journeys.digest = self._digest
        self.journeys.take_notes = self._take_notes
        self.thinker = Thinker(self.session, self.journeys)
        if thinker is False:
            self.thinker.enabled = False
        self.thinker.digest = self._digest
        self.thinker.vitals = self._vitals
        self.thinker.has_notes = self.notes.has_pending
        self.thinker.take_notes = self._take_notes
        self._last_reflex = 0.0
        self._reflex_in_flight = False
        self._dead = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self.on_answer: Callable[[str, str], None] = lambda src, a: logger.info("🤖 [%s] %s", src, a[:300])
        for j in self.journeys.interrupted:
            self.notes.push(
                f'(system) A previous session was INTERRUPTED mid-journey {j.id}: "{j.goal}" ({j.iterations} step(s) done). Check journey_status, then decide whether it still makes sense.'
            )

    async def _snapshot(self) -> BodySnapshot | None:
        return BodySnapshot.from_state(await asyncio.to_thread(self.body.state))

    async def _digest(self) -> str:
        return str((await asyncio.to_thread(self.body.state)).get("digest") or "")

    async def _vitals(self) -> dict[str, Any]:
        s = await asyncio.to_thread(self.body.state)
        return {"health": s.get("health"), "food": s.get("food"), "foodPortions": s.get("foodPortions")}

    def _take_notes(self) -> str:
        a = self.notes.take_audited()
        if a.delivered:
            logger.info("🗒 %d note(s) handed to the mind%s", a.delivered, f", {a.stamped} stamped with age" if a.stamped else "")
        if a.perished:
            logger.info("🗒 %d note(s) perished unread — %s", a.perished, a.sources)
        return a.text

    async def run(self, text: str, source: str = "you") -> str | None:
        self.thinker.touch()
        note = self._take_notes()
        self.body.set_busy(True)
        try:
            answer = await self.session.ask(f"{note}\n\n{text}" if note else text)
            self.on_answer(source, answer)
            return answer
        except Exception as err:  # noqa: BLE001 - one bad turn must not kill the process
            logger.error("❌ %s", err)
            return None
        finally:
            if self.session.busy == 0:
                self.body.set_busy(False)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _reflex(self, name: str, prompt: str) -> None:
        now = time.monotonic() * 1000
        if cfg.reflex.disabled or self._reflex_in_flight or now - self._last_reflex < cfg.reflex.cooldown_ms:
            return
        self._last_reflex = now
        self._reflex_in_flight = True
        logger.info("⚡ reflex: %s", name)

        async def go() -> None:
            try:
                await self.run(prompt, source=f"reflex:{name}")
            finally:
                self._reflex_in_flight = False

        self._spawn(go())

    def handle(self, e: SidecarEvent) -> None:
        if e.kind in ("note", "revive"):
            self.notes.push(e.text)
        elif e.kind == "reflex":
            self._reflex(e.name or "reflex", e.text)
        elif e.kind == "chat":
            self._spawn(self.run(e.text, source=f"chat:{e.name}"))
        elif e.kind == "peer":
            logger.info("🤖 <%s> %s (peer bot — logged, never answered)", e.name, e.text)
        elif e.kind == "gave_up":
            self._dead = True
            self.thinker.stop()
        if e.kind == "death":
            self._last_reflex = 0.0

    async def pump(self, stop: asyncio.Event | None = None) -> None:
        """Drain events until ``stop`` or the body gives up."""
        while not self._dead and not (stop and stop.is_set()):
            try:
                events = await asyncio.to_thread(self.body.events, cfg.sidecar.poll_ms)
            except Exception as err:  # noqa: BLE001 - the sidecar restarting is survivable
                logger.warning("event poll failed: %s", err)
                await asyncio.sleep(2)
                continue
            for e in events:
                self.handle(e)
