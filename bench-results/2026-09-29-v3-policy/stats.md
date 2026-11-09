| arm | run | task | n | success | fail | decisions | decide s | elapsed s | decide % |
|---|---|---|---|---|---|---|---|---|---|
| jev | 2026-09-29-v3-policy | logs16 | 3 | 3 | 0 | 5 (4–5) | 1.2 (1.0–1.2) | 144 (109–170) | 0.8% |
| jev | 2026-09-29-v3-policy | wood_tools | 3 | 2 | 1 | 4 (4–12) | 0.9 (0.9–3.0) | 38 (31–498*) | 0.9% |
| jev | 2026-09-29-v3-policy | cobble8 | 3 | 3 | 0 | 6 (6–7) | 1.4 (1.4–1.6) | 90 (87–96) | 1.6% |
| jev | 2026-09-29-v3-policy | iron1 | 3 | 1 | 2 | 34 (15–48*) | 9.2 (3.6–12.6) | 606 (228–905*) | 1.5% |
| jev | 2026-09-29-v3-policy | night | 3 | 2 | 1 | 102 (36–249) | 31.3 (10.4–90.3) | 623 (575–702*) | 7.0% |
| kev | 2026-09-29-v3-policy | logs16 | 3 | 3 | 0 | 4 (3–4) | 1.8 (1.4–1.8) | 85 (57–128) | 1.9% |
| kev | 2026-09-29-v3-policy | wood_tools | 3 | 2 | 1 | 4 (4–14) | 1.9 (1.8–7.9) | 64 (56–492*) | 1.9% |
| kev | 2026-09-29-v3-policy | cobble8 | 3 | 0 | 3 | 32 (32–32*) | 23.4 (21.8–33.4) | 430 (325–550) | 6.0% |
| kev | 2026-09-29-v3-policy | iron1 | 3 | 0 | 3 | 48 (48–48*) | 46.4 (45.9–52.1) | 674 (546–696) | 7.5% |
| kev | 2026-09-29-v3-policy | night | 3 | 0 | 3 | 43 (27–56) | 42.9 (25.4–55.9) | 676 (665–700*) | 6.1% |
| clm | 2026-09-29-v3-policy | logs16 | 3 | 3 | 0 | 3 (3–4) | 1.7 (1.6–2.1) | 81 (72–104) | 2.1% |
| clm | 2026-09-29-v3-policy | wood_tools | 3 | 1 | 2 | 24 (4–24*) | 10.1 (2.2–10.9) | 133 (29–157) | 7.3% |
| clm | 2026-09-29-v3-policy | cobble8 | 3 | 0 | 3 | 26 (23–32*) | 14.4 (14.3–16.2) | 611 (134–612*) | 3.3% |
| clm | 2026-09-29-v3-policy | iron1 | 3 | 0 | 3 | 48 (48–48*) | 25.3 (24.6–25.9) | 738 (283–761) | 4.3% |
| clm | 2026-09-29-v3-policy | night | 3 | 2 | 1 | 105 (104–118) | 44.3 (41.9–55.7) | 580 (577–660*) | 7.8% |

Cells: median (min–max) per episode. `~` decide time estimated as calls × p50 (rows without exact timing).
`≈` elapsed estimated as wall − 38 s setup (rows without loop_s). `*` on a max: an episode hit that budget.
Budgets (step cap/wall-clock cap): logs16 20/420s, wood_tools 24/480s, cobble8 32/600s, iron1 48/900s, night ∞/660s.
