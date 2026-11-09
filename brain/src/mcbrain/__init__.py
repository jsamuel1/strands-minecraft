"""The Python brain for strands-minecraft.

The body (mineflayer, reflexes, senses, the 54 game tools) lives in the Node sidecar
(``src/sidecar.ts``). This package is everything that decides:

- ``config``   every tunable, one registry, env override plus the reason for each default
- ``models``   provider factory: Bedrock (Sonnet 5), Bedrock Mantle (GPT Terra), TypeSafe
               System One (Jev, Kev, CLM), CLM on SageMaker (SigV4)
- ``sidecar``  MCP + event-stream client for one body
- ``session``  fork/fold shared history, window trim, throttle retry, pair audit
- ``doctor``   provider-error classification and history audit/heal
- ``notes``    NoteQueue: notes ride in front of the next turn, never mid-turn
- ``journeys`` long goals with ``[JOURNEY_DONE]`` / ``[WAITING: …]`` sentinels and the Δ critic
- ``thinker``  idle mind / journey supervisor
- ``policy``   the System One brain: a decision model picks the next macro-action
"""

__version__ = "0.1.0"
