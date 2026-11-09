# v3 run 2: harness c7e45cb, 75 episodes (2026-09-28 15:50–22:52Z)

Sonnet 5 and Terra rows here are the v3 agent results; agent arms never receive the policy state.

The Jev, Kev and CLM rows are superseded by the policy rerun at harness 767379d (048035b dropped empty containers; 767379d added the Hobson arm on top) (bench-results/2026-09-29-v3-policy). One defect affected them:
every step-1 state carried empty containers ({} / []). CLM chose `wait` from step 1 in 9/9
wood_tools/cobble8/iron1 episodes. Replaying the exact recorded requests with `empties_ablation.py` and
without the empties, it acted 9/9, and Jev/Kev kept their pick 23/24. The ablation output was retained
outside this publication branch. The rows are kept, not discarded.

1 infra failure: jev night ep2, `docker run` failed at world start (0 steps).
