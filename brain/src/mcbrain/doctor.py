"""What the provider actually complained about, and what our transcript looked like when it did.

Port of ``src/history-doctor.ts`` (issue #14). ``classify_provider_error`` names the class
and whether a retry is worth anything; ``history_shape`` renders the transcript as roles plus
tool-pair ids and nothing else; ``audit_history`` turns the shape into a verdict;
``heal_empty_text`` removes the one defect that bricks every later turn (issue #39).

Python histories are plain dicts, so the TS ``healHistory`` (rehydrating plain objects into
Message instances) has no twin here: there is nothing to rehydrate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ErrorKind = Literal["validation", "history-empty-text", "throttling", "transient", "local", "unknown"]

EMPTY_TEXT_BLOCK_ERROR = re.compile(r"text content blocks? must be non-?empty|content blocks? must not be empty", re.I)
_LOCAL = (TypeError, AttributeError, KeyError, NameError, IndexError)


@dataclass(frozen=True)
class ProviderError:
    name: str
    kind: ErrorKind
    retryable: bool
    message: str
    status: int | None = None


def classify_provider_error(err: BaseException | Any) -> ProviderError:
    """Name the class and decide if a retry is worth anything. Only throttles and transients are."""
    message = str(err)
    name = type(err).__name__ if isinstance(err, BaseException) else "Error"
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    status = status if isinstance(status, int) else None
    response = getattr(err, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, Mapping) else None
    hay = f"{name} {code or ''} {message}"
    kind: ErrorKind
    if isinstance(err, _LOCAL):
        # A TypeError never crossed a network: blaming the provider for it costs hours.
        kind = "local"
    elif EMPTY_TEXT_BLOCK_ERROR.search(message):
        kind = "history-empty-text"
    elif re.search(r"Throttl|TooManyRequests|rate ?limit|quota|429|overloaded|529", hay, re.I) or status in (429, 529):
        kind = "throttling"
    elif re.search(r"Validation|malformed|invalid.*(message|conversation)|toolResult|tool_use", hay, re.I):
        kind = "validation"
    elif (
        re.search(
            r"Stream ended without completing|ModelStreamError|incomplete (message|response)|stream (was )?(interrupted|closed|ended|reset)",
            hay,
            re.I,
        )
        or re.search(r"ServiceUnavailable|InternalServer|Timeout|ECONN|Connection(Reset|Error)|EPIPE|503|500|502", hay, re.I)
        or status in (500, 502, 503)
    ):
        kind = "transient"
    else:
        kind = "unknown"
    # ModelThrottledException is strands' own spelling of "throttled".
    if name == "ModelThrottledException":
        kind = "throttling"
    return ProviderError(name=code or name, kind=kind, retryable=kind in ("throttling", "transient"), message=message, status=status)


def _blocks(m: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return m.get("content") or []


def _short(tid: str | None) -> str:
    return tid[-6:] if tid else "?"


def is_empty_text_block(b: Mapping[str, Any]) -> bool:
    t = b.get("text")
    return isinstance(t, str) and not t.strip()


def history_shape(msgs: Sequence[Mapping[str, Any]]) -> str:
    """``u``/``a`` per role, ``[use:abc123 dig]`` / ``[res:abc123]`` per tool block, ``+t``/``+∅`` for text."""
    out = []
    for m in msgs:
        tags = []
        for b in _blocks(m):
            if "toolUse" in b:
                tu = b["toolUse"]
                tags.append(f"use:{_short(tu.get('toolUseId'))} {tu.get('name', '')}".rstrip())
            elif "toolResult" in b:
                tags.append(f"res:{_short(b['toolResult'].get('toolUseId'))}")
        texts = [b for b in _blocks(m) if "text" in b]
        text = ("+t" if any(not is_empty_text_block(b) for b in texts) else "") + (
            "+∅" if any(is_empty_text_block(b) for b in texts) else ""
        )
        role = {"assistant": "a", "user": "u"}.get(m.get("role", ""), m.get("role", "?"))
        out.append(f"{role}{'[' + ' '.join(tags) + ']' if tags else ''}{text}")
    return " ".join(out)


@dataclass(frozen=True)
class HistoryAudit:
    ok: bool
    problems: list[str]
    shape: str
    messages: int
    tool_uses: int
    tool_results: int


def audit_history(msgs: Sequence[Mapping[str, Any]]) -> HistoryAudit:
    """Everything a provider will reject about a message list, checked locally."""
    problems: list[str] = []
    use_at: dict[str, int] = {}
    res_at: dict[str, int] = {}
    uses = results = 0
    for i, m in enumerate(msgs):
        for b in _blocks(m):
            if "toolUse" in b and b["toolUse"].get("toolUseId"):
                uses += 1
                tid = b["toolUse"]["toolUseId"]
                if tid in use_at:
                    problems.append(f"duplicate toolUse {_short(tid)} (messages {use_at[tid]} and {i})")
                else:
                    use_at[tid] = i
            elif "toolResult" in b and b["toolResult"].get("toolUseId"):
                results += 1
                tid = b["toolResult"]["toolUseId"]
                if tid in res_at:
                    problems.append(f"duplicate toolResult {_short(tid)} (messages {res_at[tid]} and {i})")
                else:
                    res_at[tid] = i
    for tid, at in use_at.items():
        r = res_at.get(tid)
        if r is None:
            problems.append(f"toolUse {_short(tid)} at message {at} has NO toolResult (split pair)")
        elif r < at:
            problems.append(f"toolResult {_short(tid)} at message {r} precedes its toolUse at {at} (out-of-order fold)")
    for tid, at in res_at.items():
        if tid not in use_at:
            problems.append(f"orphan toolResult {_short(tid)} at message {at} (trim cut through a pair)")
    empty = [i for i, m in enumerate(msgs) if any("text" in b and is_empty_text_block(b) for b in _blocks(m))]
    if empty:
        problems.append(
            f"message(s) {', '.join(map(str, empty[:6]))} carry an EMPTY text block — the provider refuses the whole history (issue #39)"
        )
    if msgs and msgs[0].get("role") != "user":
        problems.append(f"history starts with '{msgs[0].get('role')}', providers require a user message first")
    return HistoryAudit(
        ok=not problems,
        problems=problems,
        shape=history_shape(msgs),
        messages=len(msgs),
        tool_uses=uses,
        tool_results=results,
    )


def diagnose(err: BaseException | Any, msgs: Sequence[Mapping[str, Any]]) -> str:
    """One log-ready block: what failed, and what we sent when it did."""
    p = classify_provider_error(err)
    a = audit_history(msgs)
    if p.kind == "local":
        verdict = "this is OUR defect, raised inside this process — the provider never saw it"
    elif p.kind == "history-empty-text":
        verdict = (
            "OUR transcript carries an empty text block: " + "; ".join(a.problems)
            if not a.ok
            else "the provider rejected an empty text block no longer in the window (issue #39)"
        )
    elif a.ok:
        verdict = (
            "history looks WELL-FORMED yet the provider called it invalid — suspect a request-level field"
            if p.kind == "validation"
            else "history is well-formed — this was the provider, not our transcript"
        )
    else:
        verdict = "history is MALFORMED: " + "; ".join(a.problems)
    return "\n".join(
        [
            f"[session] {p.name}{f' ({p.status})' if p.status else ''} kind={p.kind} retryable={str(p.retryable).lower()}: {p.message[:400]}",
            f"[session] {a.messages} messages, {a.tool_uses} toolUse / {a.tool_results} toolResult — {verdict}",
            f"[session] shape: {a.shape}",
        ]
    )


def heal_empty_text(msgs: Any) -> tuple[list[int], list[int]]:
    """Strip empty text blocks IN PLACE; drop a message that would be left blockless.

    Nothing is ever substituted for the missing words — a placeholder would tell the model it
    said something it never said. Returns ``(stripped, removed)`` indices, before removal.
    """
    stripped: list[int] = []
    removed: list[int] = []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        content = m.get("content")
        if not isinstance(content, list):
            continue
        keep = [b for b in content if not ("text" in b and is_empty_text_block(b))]
        if len(keep) == len(content):
            continue
        if not keep:
            del msgs[i]
            removed.append(i)
        else:
            m["content"] = keep
            stripped.append(i)
    return list(reversed(stripped)), list(reversed(removed))
