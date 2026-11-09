# Smoke run — SDK-native harness, strands-decider on CPU (2026-11-09)

**Harness:** `sdk-native-v1` at `3689192` (branch `feat/sdk-native-decider`), `@strands-agents/sdk` 1.24.0.
This is NOT a v3 result and not a benchmark cohort. It is a smoke test of the new
harness: n = 1 per task, two tasks, one arm. Do not compare it with
`bench-results/RESULTS-v3.md` as a result.

`run.json` is the manifest written by harness `3689192` during the run.
Later harness versions write more provenance fields, but this historical smoke
manifest stays byte-for-byte as produced. Image digest remains recorded in each
immutable episode row and summarized below.

- Arm `decider`: `SelfHostedDecisionModel` → `strands-decider serve
  StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu` (strands-decider
  0.3.0 from PyPI, torch 2.15.0+cpu, transformers 5.22.2), on the same host
  (Intel Xeon 6975P-C, 32 vCPU, no GPU). Context profile `decider` (7,500 state
  chars, 12 history steps). Option descriptions sent as choice criteria.
- World: `itzg/minecraft-server@sha256:8d0500eb…bcecf` (recorded per row as
  `mc_image`), 1.21.11, seed 8675309, tmpfs, fresh per episode.
- Cost: none beyond the local host. No paid endpoint.

| task | success | decisions | decide s | loop s | setup s | tool calls | actions (world verdict) |
|---|---|---|---|---|---|---|---|
| wood_tools | yes | 5 | 22.2 | 52.8 | 34.3 | 57 | explore_north, fell_tree_0, craft_crafting_table, place_crafting_table, craft_wooden_pickaxe — all `done` |
| logs16 | yes | 6 | 44.6 | 228.0 | 33.3 | 72 | explore_east, fell_tree_0 `done`, fell_tree_0 **`no_effect`**, fell_tree_0 ×3 `done` |

0 deaths, 0 infra errors. Decide time is CPU inference; a GPU decider would
change the decide % entirely.

The logs16 step 3 row shows the world check working: `dig_vein` returned
normally and said "Dug 6 block(s) of oak_log", but no log reached the bag (the
bag gained one oak_sapling). The verdict is `no_effect` with reason "nothing
reached the bag", `transport_ok` is true, and the next request told the model
`verdict: no_effect`.

8 of the 11 answers had a probability sum of 0.9999 or 1.0001 (per-decision
`prob_sum`), which the SDK 1.24 System One parser would have rejected.

Earlier same-seed runs on this branch (identical policy and model code) took 5
and 8 decisions for wood_tools and 4–5 for logs16. The seed fixes the terrain;
mob movement and server tick timing still vary between episodes, so one episode
per task says nothing about variance.

Files: `episodes.jsonl` (one row per episode, v3 field names, every decision
with its exact request), `run.json` (resolved world/config/budgets plus argv),
`REPORT.md` (`npm run bench -- report`).
