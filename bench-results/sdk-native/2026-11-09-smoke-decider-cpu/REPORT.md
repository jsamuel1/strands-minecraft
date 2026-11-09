| arm | run | task | n | success | fail | decisions | decide s | elapsed s | decide % |
|---|---|---|---|---|---|---|---|---|---|
| decider | 2026-11-09-smoke-decider-cpu | logs16 | 1 | 1 | 0 | 6 (6–6) | 44.6 (44.6–44.6) | 228 (228–228) | 19.6% |
| decider | 2026-11-09-smoke-decider-cpu | wood_tools | 1 | 1 | 0 | 5 (5–5) | 22.2 (22.2–22.2) | 53 (53–53) | 42.0% |

Cells: median (min–max) per episode. `*` on a max: an episode hit that budget.
Budgets (step cap/wall-clock cap): logs16 20/420s, wood_tools 24/480s, cobble8 32/600s, iron1 48/900s, night ∞/660s.
Harness: sdk-native-v1@3689192 sdk 1.24.0.
