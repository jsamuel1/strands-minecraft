# Can a System One decision model drive a long-horizon agent? Minecraft, 6 models, 90 episodes

**Short answer: not on its own, yet.** Decision models (Jev, Kev, CLM, Hobson) match LLM agents on
one-step goals, at ~10× lower latency and a fraction of the cost. Each extra step of planning
(craft a table, then a pickaxe, then mine) costs them success, while the two LLM agents went 15/15.
The best decision model, Jev, went 11/15 and was the only one to reach iron (1/3). Deciding time is
not the bottleneck; the decision models lose because they pick the wrong action for a multi-step goal
(waiting, over-gathering, wandering), not because they pick slowly.

## The question

Can a classifier-style "System One" model (one typed `Choice` per step: "which action best advances the
goal?") run a Minecraft body end to end, and where does it hand off to an LLM?

## Setup

- **Harness.** [strands-minecraft](https://github.com/cagataycali/strands-minecraft) with a Python brain
  (strands-py) driving a Node mineflayer body over MCP. The original measurement branch was
  `feat/bench-v3`: decision-model rows used harness commit `767379d`; agent rows used `c7e45cb`, which
  differs only in what the *decision* models are shown (see Caveats). This public branch is a scrubbed
  squash, so those original harness SHAs remain provenance labels in the raw rows but do not resolve as
  commits on the fork.
- **World.** Minecraft 1.21.11 (itzg), seed `8675309`, a fresh world on tmpfs and a fresh body per episode,
  one bot per server. Reflexes that move or collect on their own (item_magnet, elbow_room, idle_staring) are
  off for every arm; the remaining safety reflexes are on for every arm.
- **Two kinds of arm.**
  - *Decision models* get one `Choice` per step over code-built, legal macro-actions ("fell the oak tree at
    (18,69,76)", "craft a wooden_pickaxe", "explore east", "wait", …), plus a `last_step_succeeded` YesNo.
    The state they judge is the body's vitals, inventory, nearby resources, the last N steps (each with a
    world-checked verdict and the body state before it), events, and an action tally.
  - *LLM agents* call the body's MCP tools directly, up to 4 model turns per step.
- **Tasks** (goals verified from body state, never from the model's word):

  | task | goal | step / wall-clock budget |
  |---|---|---|
  | logs16 | collect 16 logs | 20 / 420 s |
  | wood_tools | craft a crafting table and a wooden pickaxe | 24 / 480 s |
  | cobble8 | get a pickaxe and mine 8 cobblestone | 32 / 600 s |
  | iron1 | obtain one iron ore (needs a stone pickaxe) | 48 / 900 s |
  | night | survive a night without dying | none / 660 s |

  *Diamonds and hardcore survival were not run;* iron1 is the longest chain tested (≈6 dependent steps).
- **Models**, n = 3 episodes per task (15 per model, 90 scored):

  | model | kind | served from |
  |---|---|---|
  | Jev (jev-1.13.0) | decision | TypeSafe API |
  | Kev-4B | decision | SageMaker ml.g5.xlarge, us-west-2 |
  | CLM-v0.1-8B | decision | SageMaker ml.g5.xlarge, us-west-2 |
  | [Hobson-2B v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) ([strands-decider](https://github.com/strands-labs/strands-decider)) | decision | SageMaker ml.g5.xlarge, **us-east-1** |
  | Claude Sonnet 5 | LLM agent | Bedrock |
  | GPT-5.6 Terra | LLM agent | Bedrock Mantle |

- **Hardware.** The brain and the worlds ran on one 32-core Xeon 6975P-C (123 GB) box. Each decision
  model ran on its own GPU endpoint.

## Results

Success by task (n = 3 each):

| model | logs16 | wood_tools | cobble8 | iron1 | night | **total** |
|---|---|---|---|---|---|---|
| Sonnet 5 | 3 | 3 | 3 | 3 | 3 | **15/15** |
| Terra | 3 | 3 | 3 | 3 | 3 | **15/15** |
| Jev | 3 | 2 | 3 | **1** | 2 | **11/15** |
| CLM | 3 | 1 | 0 | 0 | 2 | **6/15** |
| Kev | 3 | 2 | 0 | 0 | 0 | **5/15** |
| Hobson | 3 | 0 | 0 | 0 | 1 | **4/15** |

Per model, across all 15 episodes:

| model | success | deaths | decisions / min (median) | p50 decision latency, client / server | $ per episode (mean) | escalations to an LLM |
|---|---|---|---|---|---|---|
| Sonnet 5 | 15/15 | 0 | 6.5 | 3,298 ms / – | 1.535 | n/a |
| Terra | 15/15 | 0 | 6.1 | 2,974 ms / – | 0.663 | n/a |
| Jev | 11/15 | 1 | 4.0 | 238 ms / – | 0.020 | 0 (none wired) |
| CLM | 6/15 | 4 | 8.3 | 440 ms / 266 ms | 0.160 | 0 (none wired) |
| Kev | 5/15 | 8 | 3.8 | 661 ms / – | 0.176 | 0 (none wired) |
| Hobson | 4/15 | 4 | 2.0 | 754 ms / 514 ms | 0.236 | 0 (none wired) |

Cost: tokens at list price for Jev, Sonnet 5 and Terra. SageMaker models are billed as instance time for
the episode (g5.xlarge, $1.408/h), so their cost depends on wall time, not on tokens.

Time spent deciding, as a share of the episode (median per task): **decision models 1–8%**, **LLM agents
16–65%**. The full per-task table (decisions, decide seconds, elapsed seconds, median (min–max), budget hits
starred) is in [`stats` below](#full-table).

## Findings

1. **One-step goals are a tie, at a fraction of the cost.** All six models went 3/3 on logs16. The
   decision models made each pick in 238–754 ms versus ~3 s for the LLMs, at $0.02–0.24 an episode versus
   $0.66–1.54.
2. **Multi-step goals are where decision models fail, and they each fail differently.**
   - Hobson kept felling trees with 67–79 logs in the bag and `craft_crafting_table` on offer
     (p ≈ 0.08–0.18) until time ran out, in all 3 wood_tools episodes.
   - Kev explored on 46–48 of 48 steps on iron1 and ended with an empty bag.
   - CLM fell back to `wait` (iron1: 47/48 steps; night: 118 waits, 4 deaths).
   - Jev was the only one to chain logs → table → pickaxes → stone → iron (once), earning the Stone Age and
     Getting an Upgrade advancements.
3. **Speed is not what's missing.** The decision models spend 1–8% of an episode deciding, so going faster
   buys almost nothing. The LLMs spend 16–65% of the episode thinking and still win every long task,
   because they choose the right next subgoal.
4. **These models are sensitive to what the state looks like, so harness details matter.**
   - Empty `{}`/`[]` fields in the state flipped CLM to `wait` on step 1 in 9/9 episodes; removing them
     fixed it 9/9 with no change for Jev or Kev.
   - In v1 the harness answered "ok" for actions that did nothing, and the models repeated them. Since v2,
     only a world-checked verdict is shown. Even so, in v2 Jev chose `build_shelter` 420× across its night
     episodes; 415 of those were refused because the shelter was already sealed.
   - A context budget sized at 4 chars/token overflowed Kev's 8,192-token row (it measured 2.1–2.7).

   All three were fixed and the runs repeated. An unfair harness can make any model look bad.
5. **Where to hand off.** On this evidence a decision model is a good per-step picker when an LLM (or code)
   has already chosen the subgoal. "Craft the table now" is in reach; "plan the iron run" is not. An
   escalation path was **not** tested here (see Caveats). Natural triggers to test: low Choice confidence,
   or a world verdict of `no_effect` repeated.

## Caveats

- **n = 3 per cell.** One episode moves a cell by 33 points. Treat differences of one success as noise.
  The total success counts (15 per model) are the more stable comparison.
- **No escalation or hybrid was measured.** "Escalations to the LLM" is 0 because none were wired; this
  is a head-to-head, not a System 1 + System 2 system.
- **Two harness commits.**
  - The decision-model rows are from `767379d`. The LLM rows are from `c7e45cb`.
  - The only difference is that `767379d` drops empty containers from the *decision* state, which the LLM
    agents never receive.
  - Hobson's 15 episodes record `767379d` (1) and `7d29c3b` (14). 7d29c3b only added the policy run's
    result files; it changed no harness code. Its image used historical pre-publication strands-decider source
    at `4f1d7d0`; that source commit is not in the current public repository, while the v19 model adapter is
    public on Hugging Face.
- **Latency isn't comparable across regions.** Hobson ran in us-east-1 (us-west-2 had no GPU capacity), so
  its *client* latency includes a cross-region hop. Server latency is comparable: CLM 266 ms, Hobson
  514 ms in-episode. Hobson's warm latency on a short state was 95 ms.
- **Per-model context differs, by design.** Each model gets the context its window allows:

  | model | state budget |
  |---|---|
  | CLM | 6k chars, 6 steps |
  | Hobson | 7.5k chars, 12 steps |
  | Kev | 14k chars, 32 steps |
  | Jev | 48k chars, 64 steps |

  A larger window is a real property of a model, but it also means each model saw a different amount of
  history.
- **The decision models got an easier interface.** The harness builds only legal, grounded macro-actions
  and runs them. The LLM agents call raw tools. The decision models' result is therefore an upper bound on
  "decision model plus our action library", not a measure of the model alone.
- **Not measured:** diamonds, hardcore survival, multiplayer, longer horizons than iron1, and fine-tuning
  any decision model on Minecraft.
- **Excluded runs.** Earlier runs (v1, v2, v3 run 1, and the v3 run 2 policy rows) are kept, each with a
  README naming its defect. None are pooled here.

## Raw data and logs

- `bench-results/2026-09-29-v3-policy/`: Jev, Kev and CLM. `episodes.jsonl` holds every decision's exact
  request, answer, world verdict and reason. Also `stats.md` and `summary.md`.
- `bench-results/2026-09-29-v3-hobson/`: Hobson.
- `bench-results/2026-09-28-v3/`: Sonnet 5 and Terra, plus the `empties_ablation.py` reproduction script.
- Earlier runs, each with its README: `2026-09-28-v3-run1-defective/`, `2026-09-26-v1/`, `2026-09-26-aborted/`;
  v2 is in the main checkout at `bench-results/2026-09-28-v2/`.
- Reproduce the table: `cd brain && mcbench report <run dirs…>`.
- Sidecar logs and the ablation output were retained outside this publication branch; the episode JSONL is
  the raw data published here.
- **Video:** none captured. The harness had no screenshot capture (Chrome is absent on the host). Screenshot
  capture is queued for the next run. `docs/img/hero-loop.mp4` is the upstream project's demo, not this
  benchmark.

## Full table

Cells are median (min–max) per episode. `*` on a max means an episode hit that task's budget.

| model | task | n | success | fail | decisions | decide s | elapsed s | decide % |
|---|---|---|---|---|---|---|---|---|
| jev | logs16 | 3 | 3 | 0 | 5 (4–5) | 1.2 (1.0–1.2) | 144 (109–170) | 0.8% |
| jev | wood_tools | 3 | 2 | 1 | 4 (4–12) | 0.9 (0.9–3.0) | 38 (31–498*) | 0.9% |
| jev | cobble8 | 3 | 3 | 0 | 6 (6–7) | 1.4 (1.4–1.6) | 90 (87–96) | 1.6% |
| jev | iron1 | 3 | 1 | 2 | 34 (15–48*) | 9.2 (3.6–12.6) | 606 (228–905*) | 1.5% |
| jev | night | 3 | 2 | 1 | 102 (36–249) | 31.3 (10.4–90.3) | 623 (575–702*) | 7.0% |
| kev | logs16 | 3 | 3 | 0 | 4 (3–4) | 1.8 (1.4–1.8) | 85 (57–128) | 1.9% |
| kev | wood_tools | 3 | 2 | 1 | 4 (4–14) | 1.9 (1.8–7.9) | 64 (56–492*) | 1.9% |
| kev | cobble8 | 3 | 0 | 3 | 32 (32–32*) | 23.4 (21.8–33.4) | 430 (325–550) | 6.0% |
| kev | iron1 | 3 | 0 | 3 | 48 (48–48*) | 46.4 (45.9–52.1) | 674 (546–696) | 7.5% |
| kev | night | 3 | 0 | 3 | 43 (27–56) | 42.9 (25.4–55.9) | 676 (665–700*) | 6.1% |
| clm | logs16 | 3 | 3 | 0 | 3 (3–4) | 1.7 (1.6–2.1) | 81 (72–104) | 2.1% |
| clm | wood_tools | 3 | 1 | 2 | 24 (4–24*) | 10.1 (2.2–10.9) | 133 (29–157) | 7.3% |
| clm | cobble8 | 3 | 0 | 3 | 26 (23–32*) | 14.4 (14.3–16.2) | 611 (134–612*) | 3.3% |
| clm | iron1 | 3 | 0 | 3 | 48 (48–48*) | 25.3 (24.6–25.9) | 738 (283–761) | 4.3% |
| clm | night | 3 | 2 | 1 | 105 (104–118) | 44.3 (41.9–55.7) | 580 (577–660*) | 7.8% |
| hobson | logs16 | 3 | 3 | 0 | 4 (3–5) | 3.7 (2.8–4.6) | 92 (81–103) | 4.0% |
| hobson | wood_tools | 3 | 0 | 3 | 14 (14–16) | 11.3 (10.9–12.2) | 505 (488–520*) | 2.3% |
| hobson | cobble8 | 3 | 0 | 3 | 20 (16–21) | 15.0 (12.1–16.0) | 612 (605–666*) | 2.3% |
| hobson | iron1 | 3 | 0 | 3 | 26 (24–32) | 20.7 (19.6–25.0) | 941 (919–974*) | 2.3% |
| hobson | night | 3 | 1 | 2 | 22 (22–32) | 22.8 (16.5–23.6) | 664 (610–686*) | 3.2% |
| sonnet5 | logs16 | 3 | 3 | 0 | 7 (7–8) | 24.5 (22.6–27.2) | 122 (95–132) | 21.3% |
| sonnet5 | wood_tools | 3 | 3 | 0 | 12 (7–24) | 38.2 (23.8–85.0) | 59 (48–137) | 60.3% |
| sonnet5 | cobble8 | 3 | 3 | 0 | 12 (12–27) | 36.3 (35.2–93.2) | 137 (129–158) | 38.9% |
| sonnet5 | iron1 | 3 | 3 | 0 | 31 (16–35) | 126.3 (53.7–128.0) | 286 (217–381) | 34.9% |
| sonnet5 | night | 3 | 3 | 0 | 68 (68–87) | 368.6 (296.9–466.6) | 581 (580–586) | 64.8% |
| terra | logs16 | 3 | 3 | 0 | 7 (4–8) | 20.7 (13.6–34.4) | 137 (108–190) | 15.8% |
| terra | wood_tools | 3 | 3 | 0 | 8 (8–10) | 31.1 (26.3–32.9) | 56 (52–67) | 51.4% |
| terra | cobble8 | 3 | 3 | 0 | 24 (12–28) | 82.5 (56.1–92.2) | 165 (118–345) | 36.8% |
| terra | iron1 | 3 | 3 | 0 | 28 (18–36) | 85.0 (74.9–115.8) | 294 (188–423) | 30.5% |
| terra | night | 3 | 3 | 0 | 72 (71–80) | 225.1 (221.5–322.7) | 588 (579–590) | 43.8% |

Budgets (step / wall-clock): logs16 20/420 s, wood_tools 24/480 s, cobble8 32/600 s, iron1 48/900 s,
night no step cap / 660 s. Elapsed = from the brain's first step to the goal or the budget, excluding the
~38 s world start.
