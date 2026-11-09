# v3 policy rerun: harness 767379d, Jev / Kev / CLM × 5 tasks × 3 (2026-09-30 20:14Z – 2026-10-01 01:20Z)

45/45 episodes, 0 infra failures, 0 provider errors.
Supersedes the policy rows of 2026-09-28-v3 (c7e45cb), which sent empty containers in the step-1 state
(see that run's README and empties_ablation.log). Agent rows (Sonnet 5, Terra) come from 2026-09-28-v3:
the agent arms never receive the policy state, so the fix doesn't affect them.
Endpoints: kev-4b and clm-v01-8b on ml.g5.xlarge, us-west-2. Jev via api.typesafe.ai.
