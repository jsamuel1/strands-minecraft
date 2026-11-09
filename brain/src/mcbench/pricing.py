"""USD prices, each with its source. Unknown → NaN (never a silent zero).

- Sonnet 5 (global cross-region): AWS Pricing API, AmazonBedrockFoundationModels, us-west-2,
  usagetype USW2_input_tokens_global_standard / USW2_output_tokens_global_standard, read 2026-09-26.
- GPT-5.6 Terra: jev-strands-report.pdf (2026-09-20) price table, global.openai.gpt-5.6-terra.
  Not found in the AWS Pricing API on 2026-09-26 (searched AmazonBedrock + FoundationModels).
- Jev: docs.typesafe.ai/models via design 0020 baseline (bills input only), retrieved 2026-09-24.
- Kev: self-hosted on this dev box's CPU; no per-token price.
- Cache tokens are billed at list input rate here (an UPPER bound): the Pricing API has no
  Sonnet 5 cache-read/write SKU in us-west-2 (checked 2026-09-26). cache_read_tokens is recorded
  per episode so a discounted price can be applied later without re-running.
- CLM and Kev on SageMaker: instance-hours during the episode at the placed type's hosting rate
  (SAGEMAKER_HOURLY), not per token.
"""

from __future__ import annotations

import math

PER_MTOK: dict[str, tuple[float, float]] = {
    "global.anthropic.claude-sonnet-5": (2.0, 10.0),
    "openai.gpt-5.6-terra": (2.0, 12.0),
    "jev": (0.042, 0.0),
}
#: SageMaker real-time hosting, USD/hour, us-west-2, AWS Pricing API (AmazonSageMaker, usagetype
#: USW2-Host:<type>). g5.xlarge read 2026-09-26; the rest 2026-09-28. Unlisted types cost "unknown".
SAGEMAKER_HOURLY: dict[str, float] = {
    "ml.g5.xlarge": 1.408,
}


def token_usd(model_id: str | None, tokens_in: int, tokens_out: int) -> float:
    key = "jev" if (model_id or "").startswith("jev") else (model_id or "")
    if key not in PER_MTOK:
        return math.nan
    pin, pout = PER_MTOK[key]
    return (tokens_in * pin + tokens_out * pout) / 1e6
