# Aborted run, 2026-09-26 (harness defect, not a model result)

32 episodes, commit 43f75e5. Aborted during cobble8 because of a harness defect, which the rerun fixes.

The policy's `place_crafting_table` placed at one fixed spot, +1x from `int(pos)`. On spawn tiles next to a trunk or leaves, that spot was occupied. The failure repeated 24× in a row, and every policy arm "chose" it because it was the only table action on offer:
- wood_tools: jev ep2, clm ep2, kev ep3, clm ep3
- cobble8: jev ep1

Fixes in the rerun:
- A ring search over 16 spots, using `math.floor` rather than `int()`.
- An action that failed twice running is withheld for one decision.
- Neither table action is offered while a table is already in reach.

The logs16 rows (15/15 success) and the agent-arm rows did not use the defective path. They are kept here unchanged for reference; the committed results are in `../2026-09-26/`.
