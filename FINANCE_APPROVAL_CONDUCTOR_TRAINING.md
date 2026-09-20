# Cost Approval Request: Theo Conductor Full Training Run

**Prepared for:** Finance Director  
**Prepared on:** September 8, 2026  
**Currency:** USD  
**Requested approval:** **$25,000 operating budget** for one full training run, subject to the cost controls below

## Executive summary

We request approval for up to **$25,000** in metered worker-model API expenditure to complete one full Theo Conductor training run.

The expected API cost is approximately **$20,000**, based on an empirical run over 202 HLE physics workflows using the same frontier worker pool. The requested $25,000 authorization includes roughly 25% contingency for variation in response length, routing, failures, and retries.

This request covers paid worker-model inference only. It does not include internal or cloud GPU charges for the trainable conductor, judge-model inference, storage, or experiment tracking. Those items are not priced in the repository and should be handled separately if they create incremental expense.

Approval should be conditional on implementing an explicit worker-token limit and a $25,000 operational stop. Without those controls, the current training path can submit much larger token allowances than those used in the empirical benchmark, creating exposure substantially above the requested budget.

## Approval requested

Approve the following:

1. **Base authorization:** Up to **$25,000** for worker-model API usage for one full training run.
2. **Required controls before launch:**
   - Set an explicit per-call worker output-token cap.
   - Monitor cumulative API cost throughout the run.
   - Alert at 50%, 75%, and 90% of the approved amount.
   - Stop new paid worker requests before cumulative expenditure exceeds $25,000.
3. **No automatic contingency expansion:** Any forecast above $25,000 requires a separate approval before continuing.

## Expected cost calculation

The default training configuration uses 200 optimizer steps and a generation batch of 256 conductor rollouts per step:

| Component | Calculation | Workflows | Estimated cost |
|---|---:|---:|---:|
| Training rollouts | 200 steps × 256 rollouts | 51,200 | $18,786 |
| Scheduled evaluation | 2 evaluations × 200 questions × 8 generations | 3,200 | $1,174 |
| **Total** |  | **54,400** | **$19,960** |

The unit-cost estimate comes from the completed HLE physics benchmark:

- 202 attempted workflows
- $74.115 in recorded worker-model charges
- Average cost of **$0.3669 per attempted workflow**

Therefore:

```text
54,400 workflows × $0.3669 per workflow = approximately $19,960
```

If only one scheduled evaluation occurs, the estimate decreases by approximately $587. If scheduled evaluations are disabled, the estimated worker cost is approximately $18,800.

## Empirical model allocation

The HLE physics plans averaged approximately 3.64 worker calls per workflow. Extrapolating that behavior produces approximately 197,800 worker calls over the full run.

| Worker model | Share of planned calls | Estimated full-run cost |
|---|---:|---:|
| GPT-5.5 | 60.3% | $13,517 |
| Claude Opus 4.8 | 34.7% | $5,900 |
| Grok 4.6 | 4.1% | $404 |
| DeepSeek V4 Pro | 0.7% | $111 |
| GLM 5.3 | 0.1% | $28 |
| **Total** | **100%** | **$19,960** |

GPT-5.5 and Claude Opus account for approximately 97% of projected worker-model expenditure. Changes in their routing frequency or response length will have the largest impact on actual cost.

## Budget scenarios

| Scenario | Full-run estimate | Interpretation |
|---|---:|---|
| Empirical expected cost | **$19,960** | Extrapolates actual HLE usage, including its observed failure behavior. |
| Requested operating budget | **$25,000** | Expected cost plus approximately 25% contingency. |
| Configured output caps reached | **$234,000** | Normal-sized inputs, but every worker response reaches its model-specific YAML output limit. |
| Configured caps plus downstream propagation | **$280,000** | Maximum worker responses are also passed into and billed as inputs to later workflow steps. |
| Current uncapped training request exposure | **Approximately $5.3 million** | Nominal output-only exposure if calls consumed the context-length values currently submitted by default. This is not a likely forecast. |

The stress scenarios are exposure calculations, not predictions. Actual models normally stop well before their maximum allowance. They demonstrate why a token allowance must not be treated as a spending control by itself.

## Why maximum-token exposure is higher than expected cost

API providers generally bill for tokens actually processed, not simply the maximum requested. The empirical $20,000 forecast is therefore based on actual token consumption rather than the theoretical request allowance.

Maximum output length still matters for two reasons:

1. **Direct output cost:** A model that continues generating incurs output-token charges until it stops or reaches the configured limit.
2. **Repeated downstream input cost:** The conductor may pass an earlier worker's response to adjudication and final-answer steps. Those generated tokens are then processed and billed again as input.

For example, if two initial solvers produce long answers, an adjudicator may receive both answers, and a final synthesis model may subsequently receive all three earlier responses. One unusually long response can therefore increase several later requests.

Under the model-specific YAML limits, the estimated maximum-output charge is approximately $4.25 per workflow. Allowing for those outputs to become downstream inputs increases the plan-shaped stress estimate to approximately $5.14 per workflow:

```text
$5.14 per workflow × 54,400 workflows = approximately $280,000
```

## Important implementation risk

The HLE benchmark explicitly used each model's configured `max_output_tokens`. The default training runner follows a different path: when no fixed `max_worker_tokens` value is supplied, it currently submits each model's full `context_length` as `max_tokens`.

That means the nominal request allowance can be:

| Worker model | YAML output limit used by benchmark | Current default training request |
|---|---:|---:|
| GPT-5.5 | 32,768 | 1,000,000 |
| Claude Opus 4.8 | 65,536 | 1,000,000 |
| Grok 4.6 | 24,576 | 500,000 |
| DeepSeek V4 Pro | 40,960 | 1,048,576 |
| GLM 5.3 | 65,536 | 1,048,576 |

The approximately $5.3 million figure assumes every call generates to those submitted context-length values. That is operationally unlikely: models normally stop earlier, providers may impose lower limits, requests may be rejected, and long dependency chains may exceed context windows. Nevertheless, the current behavior is not an acceptable hard budget boundary.

## Required cost controls

Before starting the approved run, the project owner should confirm all of the following:

- **Explicit token cap:** Supply a fixed `max_worker_tokens` value or modify training to enforce each model's configured `max_output_tokens`.
- **Spend telemetry:** Aggregate the `estimated_cost_usd` recorded for every worker response.
- **Hard operational stop:** Prevent new paid calls when the approved $25,000 amount is reached; alerts alone are insufficient.
- **Retry control:** Keep retries bounded and include successful and failed requests in the cost total.
- **Routing monitoring:** Track GPT-5.5 and Claude Opus call shares because they dominate expenditure.
- **Pilot validation:** Run a small, representative pilot after applying the token cap and update the forecast if cost per workflow differs materially from $0.3669.

A fixed cap of 8,192 or 16,384 tokens would materially reduce tail risk, but a cap alone does not guarantee that the run remains below $25,000. The cumulative spend stop is the authoritative control.

## Costs excluded from this request

The repository does not provide monetary rates for the following, so they are excluded from the $19,960 estimate and the requested worker-API authorization:

- GPU time for training and serving the Qwen conductor
- Kimi K2.6 primary judge inference
- GLM 5.2 fallback judge inference
- Model and dataset storage
- Network transfer
- Weights & Biases or other experiment-tracking charges
- Engineering labor
- Taxes, provider fees, or changes in provider pricing

The conductor launcher uses separate training and generation GPUs. If those GPUs are externally billed, their incremental cost should be added as:

```text
2 GPUs × actual runtime in hours × contracted GPU hourly rate
```

Judge inference should similarly be added if the referenced endpoints are metered rather than internally provided.

## Assumptions and limitations

- The future training run routes work similarly to the 202-question HLE physics sample.
- The run completes 200 optimizer steps.
- Each training step entails 256 generated conductor workflows.
- Evaluation occurs at steps 100 and 200, with 200 questions and eight generations per question.
- Worker pricing remains equal to the checked-in frontier configuration.
- The empirical benchmark is representative of future prompt and completion lengths.
- The estimate does not include a material increase in retry frequency.
- API costs are variable usage expenses; the $25,000 request is an authorization ceiling, not a commitment to spend the entire amount.

## Recommendation

Approve **up to $25,000** for one full training run, conditional on an explicit worker-token cap, real-time cumulative cost tracking, and a hard stop at the approved amount. Based on observed HLE physics usage, the expected worker-model cost is approximately **$20,000**. Do not begin the paid run under the current context-length fallback without these safeguards.

## Repository evidence

- Training volume and evaluation defaults: `src/theo_conductor/train.py`
- Worker cap-selection behavior: `src/theo_conductor/runner.py`
- Model prices, context lengths, and output limits: `configs/worker_pool_frontier.yaml`
- Empirical HLE cost and token usage: `outputs/hle-physics-text-0000-0201/summary.json`
- Empirical HLE routing distribution: `outputs/hle-physics/summary.jsonl`

