"""Client for one body in the Node sidecar (``src/sidecar.ts``).

- ``mcp_client()`` returns a strands ``MCPClient`` over the body's streamable-HTTP MCP
  endpoint; handed to an ``Agent`` it mounts the 54 game tools with their zod-generated
  schemas, no glue.
- ``call()`` runs one tool directly (the decision policy's macro-actions) and returns the
  decoded text/JSON plus whether the tool reported an error.
- ``events()`` long-polls the body's event ring: notes, reflex asks, verified player chat,
  revive/death/gave-up. Seq-numbered; ``missed`` says how many the ring evicted first.
- ``state()`` is the digest block plus the Δ-critic snapshot and the counters.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
from strands.tools.mcp import MCPClient

from .config import cfg


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    text: str
    data: Any

    def short(self, limit: int = 400) -> str:
        return self.text if len(self.text) <= limit else self.text[:limit] + "…"


@dataclass(frozen=True)
class SidecarEvent:
    seq: int
    at: float
    kind: str
    text: str
    name: str | None = None


def _decode(result: Any) -> ToolOutcome:
    content = result.get("content", []) if isinstance(result, dict) else []
    texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
    text = "\n".join(t for t in texts if t)
    data: Any = text
    with contextlib.suppress(ValueError, TypeError):
        data = json.loads(text)
    status = result.get("status") if isinstance(result, dict) else None
    ok = status != "error" and not (isinstance(result, dict) and result.get("isError"))
    return ToolOutcome(ok=ok, text=text, data=data)


class Body:
    """One body on the sidecar. Owns an MCP client (lazily started) and an HTTP client."""

    def __init__(self, name: str, base_url: str | None = None) -> None:
        self.name = name
        self.base_url = (base_url or cfg.sidecar.url).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(cfg.sidecar.poll_ms / 1000 + 10))
        self._mcp: MCPClient | None = None
        self._started = False
        self.after = 0
        self.missed = 0

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp/{self.name}"

    def mcp_client(self) -> MCPClient:
        """The client ``call()`` uses for the harness's own tool calls (perception, goal checks)."""
        if self._mcp is None:
            self._mcp = MCPClient(url=self.mcp_url, application_name="strands-minecraft-brain", startup_timeout=30)
        return self._mcp

    def agent_mcp_client(self) -> MCPClient:
        """A SEPARATE client for an Agent's tool list; the Agent starts and stops it itself.

        Sharing ``mcp_client()`` broke every agent episode of the v3 run (2026-09-28): the
        harness's baseline goal check had already started it, and the Agent's own start() then
        raised "the client session is currently running". The sidecar is stateless streamable
        HTTP, so two clients drive the same body.
        """
        return MCPClient(url=self.mcp_url, application_name="strands-minecraft-agent", startup_timeout=30)

    def start(self) -> Body:
        """Start the MCP session for direct calls. An Agent handed ``mcp_client()`` starts it itself."""
        if not self._started:
            self.mcp_client().start()
            self._started = True
        return self

    def close(self) -> None:
        if self._mcp is not None:
            with contextlib.suppress(Exception):  # closing anyway
                self._mcp.stop(None, None, None)
            self._mcp = None
            self._started = False
        self._http.close()

    def call(self, tool: str, **arguments: Any) -> ToolOutcome:
        self.start()
        result = self.mcp_client().call_tool_sync(
            tool_use_id=f"policy-{uuid.uuid4().hex[:10]}",
            name=tool,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=cfg.sidecar.tool_timeout_s),
        )
        return _decode(result)

    def blocks(self, positions: list[dict[str, int]]) -> list[dict[str, Any]]:
        """What occupies each position and whether it is solid (harness probe; not an MCP tool)."""
        r = self._http.post(f"/blocks/{self.name}", json={"positions": positions})
        r.raise_for_status()
        return list(r.json())

    def state(self) -> dict[str, Any]:
        r = self._http.get(f"/state/{self.name}")
        r.raise_for_status()
        return r.json()

    def events(self, wait_ms: int = 0) -> list[SidecarEvent]:
        r = self._http.get(f"/events/{self.name}", params={"after": self.after, "wait": wait_ms})
        r.raise_for_status()
        body = r.json()
        self.missed += int(body.get("missed") or 0)
        events = [
            SidecarEvent(seq=e["seq"], at=e["at"], kind=e["kind"], text=e["text"], name=e.get("name")) for e in body.get("events", [])
        ]
        if events:
            self.after = events[-1].seq
        return events

    def set_busy(self, busy: bool) -> None:
        # A missed busy flag only lets an idle reflex run; never worth a turn.
        with contextlib.suppress(httpx.HTTPError):
            self._http.post(f"/busy/{self.name}", json={"busy": busy})

    def stop(self) -> list[str]:
        r = self._http.post(f"/stop/{self.name}")
        r.raise_for_status()
        return list(r.json().get("stopped", []))

    def health(self) -> dict[str, Any]:
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()


def hire_body(name: str, base_url: str | None = None, view_distance: int | str | None = None) -> Body:
    """Ask the sidecar for a worker body (the fleet's hire)."""
    base = (base_url or cfg.sidecar.url).rstrip("/")
    body: dict[str, Any] = {"name": name}
    if view_distance is not None:
        body["viewDistance"] = view_distance
    r = httpx.post(f"{base}/bodies", json=body, timeout=60)
    r.raise_for_status()
    return Body(name, base)


def retire_body(name: str, base_url: str | None = None) -> None:
    base = (base_url or cfg.sidecar.url).rstrip("/")
    httpx.delete(f"{base}/bodies/{name}", timeout=30)
