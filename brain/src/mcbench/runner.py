"""One episode = one fresh world + one sidecar + one brain + one task. Everything is measured.

Isolation: a new itzg container per episode with its world on tmpfs (nothing survives the
episode), a fixed seed, and the sidecar started with a fresh MEMORY_DIR (no waypoints, no
death history, no journeys). One bot per server. The thinker is off, the chat and reflex-turn
rails are not wired (the task prompt is the only thing that asks), and the body's idle
reflexes that move or collect (item_magnet, elbow_room, idle_staring) are off, identically for
every arm. The safety spinal cord (lava, dying, creeper_flee, fight_back, unstuck, auto_eat,
starving, auto_armor) stays on for every arm.

Two arm kinds:
- ``policy``: a DecisionModel picks one macro-action per step (Jev, Kev, CLM, and an LLM via
  ``LLMDecisionModel`` for the same-alphabet control).
- ``agent``: an LLM with the body's 54 tools over MCP, one bounded turn per step (``limits``),
  prompted with the goal and the digest — the original harness shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mcbrain.models import SAGEMAKER_PROVIDERS, BrainSpec, context_profile, create_decision_model, create_llm
from mcbrain.policy import PolicyBrain
from mcbrain.sidecar import Body

from .pricing import SAGEMAKER_HOURLY, token_usd
from .tasks import UNCAPPED_STEPS, Task

REPO = Path(__file__).resolve().parents[3]
MC_IMAGE = os.environ.get("BENCH_MC_IMAGE", "itzg/minecraft-server:latest")
MC_VERSION = os.environ.get("BENCH_MC_VERSION", "1.21.11")
SEED = os.environ.get("BENCH_SEED", "8675309")
#: Reflexes that move the body or collect on its own — off for every arm, so a gain is the brain's.
REFLEX_MODES_OFF = "item_magnet,elbow_room,idle_staring"


@dataclass
class Episode:
    run_id: str
    arm: str
    kind: str
    model_label: str
    task: str
    episode: int
    seed: str
    mc_version: str
    started_at: str
    success: bool = False
    progress: float = 0.0
    time_to_goal_s: float | None = None
    wall_s: float = 0.0
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    deaths: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    #: Prompt-cache tokens, as the provider reported them (part of tokens_in on Bedrock/Mantle).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    server_latency_p50_ms: float | None = None
    provider_errors: dict[str, int] = field(default_factory=dict)
    #: The first message of each error class — a count alone hid Terra's real 400 once.
    error_samples: dict[str, str] = field(default_factory=dict)
    #: SageMaker arms: the instance type the endpoint was running on (describe-endpoint).
    instance_type: str | None = None
    throttles: int = 0
    answering_models: list[str] = field(default_factory=list)
    final_inventory: dict[str, int] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    #: Policy arms: every decision with its distribution head and the tools' own outcome text.
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    #: Which task budget ended a failed episode: "steps" | "wall" | None (succeeded, or infra error).
    cap_hit: str | None = None
    #: Seconds from the brain's first step to the end of the loop (goal, or a budget).
    loop_s: float | None = None
    #: Seconds spent waiting on the model inside the loop. Exact for both arm kinds from v3:
    #: policy = sum of per-decision round trips; agent = sum of per-model-call latencies.
    decide_s: float | None = None
    #: Agent arms: every model call's latency, ms (v1/v2 kept only p50/p95, so decide time was ~estimated).
    call_latencies_ms: list[float] = field(default_factory=list)
    #: Policy arms: the state budget this model was given, chars (CLM is smaller than the others).
    state_chars_budget: int | None = None
    #: Policy arms: how many aligned past steps this model may see (CLM 12, Kev 32, Jev/LLM 64).
    history_steps_budget: int | None = None
    #: Policy arms: the full context profile (budgets, history, event/outcome chars, previous_states on/off).
    context_profile: dict[str, Any] = field(default_factory=dict)
    #: Harness version: v3 adds world-checked verdicts, aligned history, the last-step question.
    harness: str = "v3"
    brain_commit: str | None = None
    strands_commit: str = "7599ed078231d14577075550966b30a1b264cfff"


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q) - 1], 1)


class World:
    """A throwaway itzg server on tmpfs. Port is per-episode so a slow teardown cannot collide."""

    def __init__(self, port: int, rcon_port: int) -> None:
        self.name = f"bench-mc-{uuid.uuid4().hex[:8]}"
        self.port = port
        self.rcon_port = rcon_port

    def __enter__(self) -> World:
        cmd = [
            "docker", "run", "-d", "--name", self.name,
            "-p", f"127.0.0.1:{self.port}:25565", "-p", f"127.0.0.1:{self.rcon_port}:25575",
            "-e", "EULA=TRUE", "-e", "TYPE=VANILLA", "-e", f"VERSION={MC_VERSION}", "-e", "ONLINE_MODE=FALSE",
            "-e", "MEMORY=2G", "-e", f"SEED={SEED}", "-e", "DIFFICULTY=easy", "-e", "MODE=survival",
            "-e", "SPAWN_PROTECTION=0", "-e", "VIEW_DISTANCE=6", "-e", "ENABLE_RCON=true", "-e", "RCON_PASSWORD=bench",
            "--tmpfs", "/data:size=3g", MC_IMAGE,
        ]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True)
        deadline = time.time() + 240
        while time.time() < deadline:
            status = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Health.Status}}", self.name], capture_output=True, text=True
            ).stdout.strip()
            if status == "healthy":
                return self
            time.sleep(3)
        raise RuntimeError(f"{self.name} never became healthy")

    def rcon(self, command: str) -> str:
        return subprocess.run(["docker", "exec", self.name, "rcon-cli", command], capture_output=True, text=True).stdout.strip()

    def __exit__(self, *exc: object) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


class Sidecar:
    """The Node body sidecar, with a fresh MEMORY_DIR and the benchmark's reflex settings."""

    def __init__(self, mc_port: int, port: int, username: str = "StrandsBot") -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.username = username
        self.mem = Path(tempfile.mkdtemp(prefix="mcbench-mem-"))
        self.log = self.mem / "sidecar.log"
        self.env = {
            **os.environ,
            "MC_HOST": "127.0.0.1", "MC_PORT": str(mc_port), "MC_USERNAME": username, "MC_VERSION": MC_VERSION,
            "SIDECAR_PORT": str(port), "MEMORY_DIR": str(self.mem), "REFLEX_MODES_OFF": REFLEX_MODES_OFF,
            "THINKER_DISABLED": "true", "VOICE_DISABLED": "true",
        }  # fmt: skip
        self.proc: subprocess.Popen[bytes] | None = None
        #: Where to keep the sidecar log after the episode (None = discard with the world).
        self.keep_log: Path | None = None

    def __enter__(self) -> Sidecar:
        with self.log.open("wb") as fh:
            self.proc = subprocess.Popen(["npx", "tsx", "src/sidecar.ts"], cwd=REPO, env=self.env, stdout=fh, stderr=subprocess.STDOUT)
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"sidecar exited: {self.log.read_text()[-800:]}")
            try:
                if self.username in httpx.get(f"{self.url}/health", timeout=2).json().get("bodies", []):
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise RuntimeError(f"sidecar never came up: {self.log.read_text()[-800:]}")

    def __exit__(self, *exc: object) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.keep_log is not None:
            self.keep_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.log, self.keep_log)  # the body's own account of the episode
        shutil.rmtree(self.mem, ignore_errors=True)


def _endpoint_instance(spec: BrainSpec) -> str | None:
    """The instance type SageMaker actually placed the endpoint on (InstancePools may fall back)."""
    import boto3

    try:
        session = (
            boto3.Session(profile_name=spec.profile, region_name=spec.region) if spec.profile else boto3.Session(region_name=spec.region)
        )
        variant = session.client("sagemaker").describe_endpoint(EndpointName=spec.endpoint)["ProductionVariants"][0]
        # With InstancePools the placed type is the pool holding instances, not a top-level field.
        placed = [p["InstanceType"] for p in variant.get("InstancePools") or [] if p.get("CurrentInstanceCount")]
        return placed[0] if placed else (str(variant.get("CurrentInstanceType") or variant.get("InstanceType") or "") or None)
    except Exception:  # noqa: BLE001 - an unknown type is recorded as unknown, never guessed
        return None


def _commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or None


def _tables_near(body: Body) -> int:
    out = body.call("find_blocks", name="crafting_table", maxDistance=8, count=4)
    return len(out.data) if isinstance(out.data, list) else 0


def _state_with_marks(body: Body, marks: dict[str, Any]) -> dict[str, Any]:
    s = body.state()
    # World truth for a placed table: more crafting tables within 8 blocks than at the start.
    # (Measured from the world, never from the model; a village table present at spawn does not count.)
    if "tablesAtStart" not in marks:
        marks["tablesAtStart"] = _tables_near(body)
    elif not marks.get("placedTable") and _tables_near(body) > marks["tablesAtStart"]:
        marks["placedTable"] = True
    s["_placedTable"] = marks.get("placedTable", False)
    t = s.get("timeOfDay") or 0
    if t >= 13000:
        marks["sawNight"] = True
    if marks.get("sawNight") and 0 <= t < 1000:
        marks["passedNight"] = True
    s["_passedNight"] = marks.get("passedNight", False)
    return s


async def _run_policy(ep: Episode, spec: BrainSpec, task: Task, body: Body, marks: dict[str, Any]) -> None:
    model = create_decision_model(spec)
    prof = context_profile(spec)
    ep.state_chars_budget, ep.history_steps_budget = prof.state_chars, prof.history_steps
    ep.context_profile = asdict(prof)
    brain = PolicyBrain(model, body, task.goal, profile=prof)
    lat: list[float] = []
    srv: list[float] = []
    t0 = time.monotonic()
    for i in range(task.max_steps):
        used = time.monotonic() - t0
        if used > task.max_wall_s:
            break
        brain.budget = {
            "steps_used": i,
            # None = no step limit (night is wall-clock bound); never a sentinel number the model must decode.
            "step_cap": task.max_steps if task.max_steps < UNCAPPED_STEPS else None,
            "seconds_used": round(used),
            "seconds_cap": task.max_wall_s,
            "progress": ep.progress,
        }
        d = await brain.step()
        ep.steps += 1
        ep.model_calls += 1
        if d.error and d.action == "error":
            name = d.error.split(":")[0]
            ep.provider_errors[name] = ep.provider_errors.get(name, 0) + 1
            ep.error_samples.setdefault(name, d.error[:300])
            if "Throttl" in d.error or "RateLimit" in d.error:
                ep.throttles += 1
        else:
            lat.append(d.latency_ms)
            if d.server_latency_ms is not None:
                srv.append(float(d.server_latency_ms))
            if d.model_id and d.model_id not in ep.answering_models:
                ep.answering_models.append(d.model_id)
        ep.tokens_in += int(d.usage.get("inputTokens", 0))
        ep.tokens_out += int(d.usage.get("outputTokens", 0))
        ep.actions.append(f"{d.action}:{d.verdict}")
        ep.decisions.append(
            {
                "step": d.step,
                "action": d.action,
                "verdict": d.verdict,
                "check": d.check,
                "reason": d.reason,
                "transport_ok": d.ok,
                "last_step_p": d.last_step_p,
                "conf": d.confidence,
                "top": d.top,
                "ms": d.latency_ms,
                "server_ms": d.server_latency_ms,
                "options": d.options,
                "state_chars": d.state_chars,
                "outcome": (d.outcome or d.error or "")[:400],
                "request": d.request,
            }
        )
        done, ep.progress = task.check(_state_with_marks(body, marks))
        if done:
            ep.success = True
            ep.time_to_goal_s = round(time.monotonic() - t0, 1)
            break
    ep.loop_s = round(time.monotonic() - t0, 1)
    ep.decide_s = round(sum(lat) / 1000, 2)
    ep.latency_p50_ms, ep.latency_p95_ms = _pct(lat, 50), _pct(lat, 95)
    ep.server_latency_p50_ms = _pct(srv, 50)
    if not ep.success:
        ep.cap_hit = "steps" if ep.steps >= task.max_steps else "wall"


async def _run_agent(ep: Episode, spec: BrainSpec, task: Task, body: Body, marks: dict[str, Any]) -> None:
    from mcbrain.brain import Brain

    brain = Brain(create_llm(spec), body, thinker=False)
    t0 = time.monotonic()
    first = (
        f"Your goal: {task.goal} This is a benchmark: the goal is verified from your inventory and body, so actually do it. "
        "Work step by step with your tools. Use the composite tools (find_blocks, dig_vein, craft_item). "
        "Do not start journeys or hire workers — just act."
    )
    for i in range(task.max_steps):
        if time.monotonic() - t0 > task.max_wall_s:
            break
        s = body.state()
        prompt = first if i == 0 else f"Status:\n{s.get('digest')}\nContinue toward the goal: {task.goal}"
        budget = max(1.0, task.max_wall_s - (time.monotonic() - t0))
        try:
            answer = await asyncio.wait_for(brain.session.agent.invoke_async(prompt, limits={"turns": 4}), budget)
            brain.session._record(brain.session.agent, answer)  # noqa: SLF001 - same accounting as ask()
        except TimeoutError:
            break
        except Exception as err:  # noqa: BLE001 - provider errors are data
            # Counted once, per model call, by the AfterModelCallEvent hook below; the smoke
            # run double-counted (20 failed turns → 40) when this branch counted too.
            ep.error_samples.setdefault(type(err).__name__, str(err)[:300])
        ep.steps += 1
        done, ep.progress = task.check(_state_with_marks(body, marks))
        if done:
            ep.success = True
            ep.time_to_goal_s = round(time.monotonic() - t0, 1)
            break
    ep.loop_s = round(time.monotonic() - t0, 1)
    if not ep.success:
        ep.cap_hit = "steps" if ep.steps >= task.max_steps else "wall"
    ep.model_calls = len(brain.stats.latencies_ms)
    ep.call_latencies_ms = [round(x, 1) for x in brain.stats.latencies_ms]
    ep.decide_s = round(sum(brain.stats.latencies_ms) / 1000, 2)
    ep.latency_p50_ms, ep.latency_p95_ms = _pct(brain.stats.latencies_ms, 50), _pct(brain.stats.latencies_ms, 95)
    for name, n in brain.stats.errors.items():
        ep.provider_errors[name] = ep.provider_errors.get(name, 0) + n
        if "Throttl" in name:
            ep.throttles += n
    for t in brain.session.turns:
        ep.tokens_in += int(t["usage"].get("inputTokens", 0))
        ep.tokens_out += int(t["usage"].get("outputTokens", 0))
        ep.cache_read_tokens += int(t["usage"].get("cacheReadInputTokens", 0))
        ep.cache_write_tokens += int(t["usage"].get("cacheWriteInputTokens", 0))
    # strands' tool_metrics.call_count is cumulative per agent, so only the LAST turn's map is the
    # episode's count (v1/v2 summed every turn's map: Sonnet 5 night ep1 read 2,077 vs 131 executed).
    last_tools = (brain.session.turns[-1].get("tools") or {}) if brain.session.turns else {}
    ep.actions = [f"{k}x{v}" for k, v in last_tools.items()]
    ep.answering_models = [spec.model_id]


async def run_episode(
    run_id: str, arm: str, kind: str, spec: BrainSpec, task: Task, n: int, ports: tuple[int, int, int], logs: Path | None = None
) -> Episode:
    ep = Episode(
        run_id=run_id,
        arm=arm,
        kind=kind,
        model_label=spec.label,
        task=task.id,
        episode=n,
        seed=SEED,
        mc_version=MC_VERSION,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        brain_commit=_commit(),
    )
    t0 = time.monotonic()
    mc_port, rcon_port, sc_port = ports
    if spec.provider in SAGEMAKER_PROVIDERS:
        ep.instance_type = _endpoint_instance(spec)
    try:
        with World(mc_port, rcon_port) as world, Sidecar(mc_port, sc_port) as sc:
            if logs is not None:
                sc.keep_log = logs / f"{task.id}.{arm}.{n}.sidecar.log"
            for cmd in task.setup:
                world.rcon(cmd)
            body = Body(sc.username, sc.url)
            marks: dict[str, Any] = {}
            _state_with_marks(body, marks)  # baseline: tables in reach before the brain acts
            try:
                await (_run_policy if kind == "policy" else _run_agent)(ep, spec, task, body, marks)
                final = body.state()
                ep.final_inventory = dict(final.get("inv") or {})
                counters = final.get("counters") or {}
                ep.tool_calls = int(counters.get("toolCalls", 0))
                ep.tool_errors = int(counters.get("toolErrors", 0))
                ep.deaths = int(counters.get("deaths", 0))
            finally:
                body.close()
    except Exception as err:  # noqa: BLE001 - an infrastructure failure is recorded, not hidden
        ep.error = f"{type(err).__name__}: {err}"[:500]
    ep.wall_s = round(time.monotonic() - t0, 1)
    if spec.provider in SAGEMAKER_PROVIDERS:
        # Instance time during the episode, at the hourly rate of the type the endpoint actually runs on.
        ep.cost_usd = round(SAGEMAKER_HOURLY.get(ep.instance_type or "", float("nan")) * ep.wall_s / 3600, 5)
        if ep.cost_usd != ep.cost_usd:
            ep.cost_usd = None
    else:
        usd = token_usd(spec.model_id, ep.tokens_in, ep.tokens_out)
        # NaN → unknown, never zero; and no model call means no bill, not a measured $0.
        ep.cost_usd = None if usd != usd or ep.model_calls == 0 else round(usd, 5)
    return ep


def append_jsonl(path: Path, ep: Episode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
