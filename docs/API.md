# External model configuration

FW4 training uses an OpenAI-compatible Chat Completions endpoint with structured JSON output. It extracts mathematical facts and classifies fact–trajectory relations. Teacher logits come from the current role policy, not the external model. Evaluation does not need these credentials or calls.

Set these environment variables, or load an ignored `.env.local` copied from `.env.example`:

| Variable | Meaning |
|---|---|
| `OPSD_API_BASE_URL` | Provider endpoint, including its API path |
| `OPSD_API_MODEL` | Model/service ID accepted by that provider |
| `OPSD_API_KEY` | Secret credential; never written to launch arguments or manifests |
| `OPSD_API_KEY_ENV` | Optional alternative credential variable name |
| `OPSD_API_TPM` | This job's allocated tokens per minute |
| `OPSD_API_RPM` | This job's allocated requests per minute |
| `OPSD_API_CONCURRENCY` | Maximum simultaneous extraction/relation requests per stage |
| `OPSD_API_MAX_TOKENS` | Per-response output limit, default 4096 |

Actual training refuses empty required settings. Dry-run only prints a command and does not establish API access. No company endpoint, service alias or quota is assumed. The previous provider's internal model alias is not a reproducible public model identifier; choose a suitable accessible model and record the exact ID. A provider change can change extracted facts and training results.

Extraction and relation attempts share a process-local rolling limiter in this release. Output reservations and retries matter. SDK retries are disabled on rate-limited calls so they do not silently bypass accounting; application retries remain bounded. This is operational hardening relative to the historical snapshot, not a claim that historical training used the same limiter. Other processes, servers and users remain outside this limiter and must divide the account quota. Token estimates and provider accounting can differ, so leave an appropriate provider-specific margin.

A provider can return invalid JSON without returning a rate-limit error. A one-character `{` ending with `finish_reason=stop` should be investigated as malformed output, not automatically treated as insufficient output length. Inspect extraction success, JSON/schema failures, logical requests and API attempts separately. Full raw API audit logging is disabled by default because it can contain training text and provider diagnostics.

The extraction and relation prompt builders remain in `verl/trainer/ppo/opd_fact_cards.py`. Their content and the mathematical filtering rules are retained from the FW4 snapshot. Do not change prompts or silently switch models midway through a result-producing run.
