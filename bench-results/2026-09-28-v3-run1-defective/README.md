# v3 run 1 (harness f0cbcc7): kept aside, two harness defects

75 episodes, 2026-09-28 10:35–15:25Z. Not the v3 results; the rerun is in 2026-09-28-v3/.

1. All 30 agent episodes (Sonnet 5, Terra) are infra failures: `MCPClientInitializationError: the client session is
   currently running`. The runner's baseline goal check started the Body's MCP client, and the Agent was
   handed the same client and tried to start it again. Latent since ec9fbde (baseline check added). v1 predates
   it; v2 ran policy arms only. Fix: the Agent gets its own client (Body.agent_mcp_client).
2. All 3 Kev night episodes: 1,087 HTTP 422 "row limit 8192" rejections. The Kev state budget (24,000 chars)
   assumed about 4 chars/token. Kev's own tokenizer (Qwen3.5-4B) measured 2.11–2.74 over 370 real v3 states,
   and night states grew past ~19.3k chars. Fix: 14,000 chars.

Jev and CLM rows are unaffected by either defect. They are kept here for comparison and are not pooled with the rerun.
