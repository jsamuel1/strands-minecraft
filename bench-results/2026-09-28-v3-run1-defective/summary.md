| arm | model (as answered) | task | n | success | progress | time-to-goal s (median) | steps | model calls | tool calls | deaths | tokens in/out | $ total | p50 / p95 ms | server p50 ms | errors | infra fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | logs16 | 3 | 1/3 | 0.33 | 102 | 20 | 20 | 233 | 0 | 132,204 / 4,318 | 0.0056 | 249 / 276 | – | 0 | 0 |
| jev | jev-1.13.0 | wood_tools | 3 | 3/3 | 1.00 | 47 | 4 | 4 | 45 | 0 | 17,873 / 1,308 | 0.0008 | 261 / 298 | – | 0 | 0 |
| jev | jev-1.13.0 | cobble8 | 3 | 3/3 | 1.00 | 87 | 7 | 7 | 98 | 0 | 38,174 / 2,334 | 0.0016 | 264 / 302 | – | 0 | 0 |
| jev | jev-1.13.0 | iron1 | 3 | 1/3 | 0.33 | 387 | 48 | 48 | 536 | 0 | 755,587 / 12,544 | 0.0317 | 274 / 386 | – | 0 | 0 |
| jev | jev-1.13.0 | night | 3 | 3/3 | 1.00 | 579 | 53 | 53 | 578 | 0 | 2,494,467 / 24,756 | 0.1048 | 299 / 644 | – | TypeSafeAPITimeoutError×5, TypeSafeInternalServerError×1 | 0 |
| kev | kev-latest | logs16 | 3 | 3/3 | 1.00 | 89 | 3 | 3 | 37 | 0 | 11,256 / 1,544 | 0.1586 | 390 / 725 | – | 0 | 0 |
| kev | kev-latest | wood_tools | 3 | 3/3 | 1.00 | 26 | 4 | 4 | 46 | 0 | 12,168 / 1,783 | 0.0992 | 401 / 671 | – | 0 | 0 |
| kev | kev-latest | cobble8 | 3 | 0/3 | 0.00 | – | 32 | 32 | 353 | 0 | 308,043 / 10,195 | 0.5530 | 693 / 1253 | – | 0 | 0 |
| kev | kev-latest | iron1 | 3 | 0/3 | 0.00 | – | 48 | 48 | 529 | 0 | 617,299 / 15,562 | 0.8744 | 915 / 1313 | – | 0 | 0 |
| kev | kev-latest | night | 3 | 1/3 | 0.33 | 575 | 371 | 371 | 2397 | 3 | 385,394 / 12,814 | 0.7885 | 1085 / 1531 | – | ModelError×1087 | 0 |
| clm | clm-latest | logs16 | 3 | 3/3 | 1.00 | 128 | 3 | 3 | 37 | 0 | 14,049 / 0 | 0.1991 | 500 / 732 | 284 | 0 | 0 |
| clm | clm-latest | wood_tools | 3 | 0/3 | 0.00 | – | 24 | 24 | 265 | 0 | 103,779 / 0 | 0.2047 | 412 / 458 | 264 | 0 | 0 |
| clm | clm-latest | cobble8 | 3 | 0/3 | 0.00 | – | 32 | 32 | 354 | 0 | 134,661 / 0 | 0.2671 | 411 / 462 | 263 | 0 | 0 |
| clm | clm-latest | iron1 | 3 | 0/3 | 0.00 | – | 48 | 48 | 529 | 0 | 199,930 / 0 | 0.3589 | 410 / 457 | 262 | 0 | 0 |
| clm | clm-latest | night | 3 | 2/3 | 0.67 | 580 | 104 | 104 | 1145 | 1 | 485,887 / 0 | 0.7604 | 411 / 519 | 264 | 0 | 0 |
| sonnet5 | bedrock:global.anthropic.claude-sonnet-5 | logs16 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| sonnet5 | bedrock:global.anthropic.claude-sonnet-5 | wood_tools | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| sonnet5 | bedrock:global.anthropic.claude-sonnet-5 | cobble8 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| sonnet5 | bedrock:global.anthropic.claude-sonnet-5 | iron1 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| sonnet5 | bedrock:global.anthropic.claude-sonnet-5 | night | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| terra | mantle:openai.gpt-5.6-terra | logs16 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| terra | mantle:openai.gpt-5.6-terra | wood_tools | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| terra | mantle:openai.gpt-5.6-terra | cobble8 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| terra | mantle:openai.gpt-5.6-terra | iron1 | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
| terra | mantle:openai.gpt-5.6-terra | night | 0 | 0/0 | – | – | – | – | – | 0 | 0 / 0 | unknown | – / – | – | 0 | 3 |
