# strands-minecraft brain (Python)

The Python brain for strands-minecraft, on [strands-py](https://github.com/strands-agents/harness-sdk).
No Python Minecraft client supports 1.21, so the **body stays in Node**: `src/sidecar.ts` keeps
mineflayer, the reconnect proxy, legs, the 300 ms reflexes, the sentinel, chat classification and
the 54 game tools, and exposes each body over MCP plus an event stream. This package decides.

```
 ┌──────────── Python brain (this package) ────────────┐        ┌──────── Node sidecar ────────┐
 │ models.py   bedrock · mantle · typesafe · kev · clm │  MCP   │ /mcp/<body>   54 tools (zod)  │
 │ session.py  fork/fold shared history, trim, retry   │◄──────►│ /events/<body> notes, reflex, │
 │ notes.py    NoteQueue (front of the next turn)      │  HTTP  │                chat, death    │
 │ journeys.py [JOURNEY_DONE] / [WAITING:] + Δ critic  │        │ /state/<body> digest + Δ snap │
 │ thinker.py  idle mind / journey supervisor          │        │ body · legs · reflexes ·      │
 │ policy.py   System One: code builds legal actions,  │        │ sentinel · chatrail · rails   │
 │             the decision model picks one (Choice)   │        └──────────────┬───────────────┘
 └─────────────────────────────────────────────────────┘                       │ mineflayer
                                                                        Minecraft 1.21.11
```

## Two kinds of brain

| `STRANDS_MODEL_PROVIDER` | model | brain |
|---|---|---|
| `bedrock` (default) | `global.anthropic.claude-sonnet-5` | tool-calling LLM over the 54 MCP tools + journeys |
| `mantle` | `openai.gpt-5.6-terra` (Bedrock OpenAI-compatible endpoint) | tool-calling LLM |
| `typesafe` | `jev-latest` (`TYPESAFE_API_KEY`) | System One **policy** |
| `kev` | `kev-latest` at `KEV_BASE_URL` (default `http://127.0.0.1:8009`) | System One policy |
| `clm` | `clm-latest` at `CLM_BASE_URL` (clm-serve, same wire) | System One policy |
| `clm-sagemaker` | CLM-8B behind SageMaker `InvokeEndpoint` (`CLM_ENDPOINT`, `CLM_REGION`) | System One policy |

A System One model returns typed judgments, not tool calls. So `policy.py` keeps **facts in code**
(what is legal and grounded now: the tree `find_blocks` saw, the craft the bag can pay for, no iron
offered to a wooden pickaxe, creepers fled never punched) and asks **one `Choice`** over those
macro-actions. Code then runs the chosen macro-action as one to three sidecar tool calls. An LLM
can sit in the same seat (`LLMDecisionModel`) for a same-alphabet comparison.

There is no "GPT-6 Terra" on Bedrock (checked 2026-09-26 in us-west-2 and us-east-2: the gpt-6
line is astra/sol/luna and only `openai.gpt-6-astra` is invocable on-demand). The Terra model is
`openai.gpt-5.6-terra`.

## Run

```bash
cd brain && uv sync --extra dev                  # strands-py pinned to #4559 (decisions + TypeSafe)
(cd .. && MC_HOST=… npm run sidecar)             # the body, on :3010
STRANDS_MODEL_PROVIDER=bedrock  uv run strands-minecraft-brain            # REPL + rails
STRANDS_MODEL_PROVIDER=typesafe uv run strands-minecraft-brain --goal "collect 16 logs" --steps 20
```

## Benchmark

```bash
uv run mcbench run --arms jev,kev,clm,sonnet5,terra --tasks logs16,wood_tools,cobble8 --episodes 3 --out ../bench-results/<date>
uv run mcbench summarize ../bench-results/<date>/episodes.jsonl
```

Each episode gets a fresh itzg server (fixed seed `8675309`, 1.21.11, world on tmpfs), a fresh
sidecar (empty `MEMORY_DIR`), and one bot. The thinker is off, and the idle reflexes that move or
collect by themselves (`item_magnet, elbow_room, idle_staring`) are off for every arm. Goals are
**verified from body state**, never from the model's words. Recorded per episode: success,
progress, time to goal, steps, model calls, tool calls and errors, deaths, tokens, cost (with the
price source in `mcbench/pricing.py`), client latency p50/p95, CLM server latency, provider errors
and throttles, the exact model id that answered, and the action trail.

CLM bills as SageMaker instance time. Create it before a run and delete it after:
`../infra/clm-sagemaker/deploy.sh deploy|delete`.

## Develop

```bash
uv run pytest && uv run ruff check src tests && uv run mypy src
```

Tunables live only in `src/mcbrain/config.py` (env override plus a reason for each default), as in
`src/config.ts`.
