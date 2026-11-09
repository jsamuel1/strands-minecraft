| arm | run | task | n | success | fail | decisions | decide s | elapsed s | decide % |
|---|---|---|---|---|---|---|---|---|---|
| hobson | 2026-09-29-v3-hobson | logs16 | 3 | 3 | 0 | 4 (3–5) | 3.7 (2.8–4.6) | 92 (81–103) | 4.0% |
| hobson | 2026-09-29-v3-hobson | wood_tools | 3 | 0 | 3 | 14 (14–16) | 11.3 (10.9–12.2) | 505 (488–520*) | 2.3% |
| hobson | 2026-09-29-v3-hobson | cobble8 | 3 | 0 | 3 | 20 (16–21) | 15.0 (12.1–16.0) | 612 (605–666*) | 2.3% |
| hobson | 2026-09-29-v3-hobson | iron1 | 3 | 0 | 3 | 26 (24–32) | 20.7 (19.6–25.0) | 941 (919–974*) | 2.3% |
| hobson | 2026-09-29-v3-hobson | night | 3 | 1 | 2 | 22 (22–32) | 22.8 (16.5–23.6) | 664 (610–686*) | 3.2% |

Cells: median (min–max) per episode. `~` decide time estimated as calls × p50 (rows without exact timing).
`≈` elapsed estimated as wall − 38 s setup (rows without loop_s). `*` on a max: an episode hit that budget.
Budgets (step cap/wall-clock cap): logs16 20/420s, wood_tools 24/480s, cobble8 32/600s, iron1 48/900s, night ∞/660s.
