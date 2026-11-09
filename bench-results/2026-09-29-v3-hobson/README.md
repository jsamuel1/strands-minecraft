# v3 Hobson arm: strands-decider / hobson-2b-v19 × 5 tasks × 3 (2026-10-01 01:20Z – 03:51Z)

15/15 episodes, 0 infra failures, 0 provider errors. Endpoint hobson-2b-v19 on ml.g5.xlarge in **us-east-1**
(us-west-2 had InsufficientInstanceCapacity twice), so client latency includes the cross-region hop; server
latency (hobson's own timing) does not. Image sha256:27d039bd…, historical pre-publication source @ 4f1d7d0,
base Qwen3.5-2B-Base @ b1485b2f. Current source is [strands-decider](https://github.com/strands-labs/strands-decider),
and the v19 adapter is [public on Hugging Face](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19);
the historical source commit is not in the public repository.
brain_commit reads 767379d (1 episode) and 7d29c3b (14): 7d29c3b only added the policy run's result files,
0 lines of harness code changed, so all 15 ran the identical harness.
