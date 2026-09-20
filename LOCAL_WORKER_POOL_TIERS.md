# Theo Conductor worker-pool tiers

**Status:** proposed experiment design  
**Updated:** 2026-09-09  
**Scope:** worker inference only; conductor training/serving and judge inference are not included

## Recommendation

Use a seven-model local pool on four 141 GB H200s. Seven matches the checked-in
frontier tier, while the mix below gives the conductor meaningful choices among
deliberate reasoning, general instruction following, code, multimodal work,
critique, and fast verification.

Run the first proof with the highest-fidelity publisher checkpoints (BF16 where
available). The seven models contain about 161 billion parameters in total, or
roughly 321 GB if every weight were BF16. Four H200s provide
564 GB of aggregate HBM, so even that conservative weight estimate fits while
leaving roughly 243 GB in aggregate for KV cache, CUDA graphs, activations, and
serving overhead. This is a capacity estimate rather than a promise about a
particular serving build; measure actual residency before the benchmark.

Then repeat a representative slice with weight-only 4-bit quantization. Use that
as an efficiency/sensitivity result, not as the sole result: otherwise a gain or
loss may be attributable to quantization rather than orchestration.

## Tier comparison

This is the canonical side-by-side table. Add one row per future price/compute
tier and keep the column meanings fixed.

| Tier ID | Worker substrate | Resident workers | Precision | Served context cap | Max workflow calls | Marginal price | Primary question |
|---|---|---:|---|---:|---:|---|---|
| `frontier-api` | Metered APIs | 7 | Provider native | Model-specific, 500K-1M configured | 7 | USD/token | Does Theo improve quality over strong frontier workers? |
| `local-4xh200-hifi` | 4 x H200 141 GB | 7 | Publisher high-fidelity checkpoint; BF16 where available | 16K initially | 7 | GPU-hours | Does routing/composition help among affordable open-weight workers? |
| `local-4xh200-int4` | 4 x H200 141 GB | 7 | AWQ/GPTQ 4-bit where validated | 32K target | 7 | GPU-hours | How much throughput can be gained without erasing the orchestration benefit? |
| `<new-tier-id>` | `<hardware or API>` | `<n>` | `<precision>` | `<tokens>` | `<n>` | `<unit>` | `<hypothesis>` |

Do not compare the API tier's dollar-per-token directly with a local tier until
an accounting rate has been supplied. The portable units are accuracy,
workflows/hour, generated tokens, calls/workflow, and allocated GPU-hours.

## Proposed seven-worker local pool

| Worker ID | Checkpoint | Params | Native context | Assignment in Theo | Why it earns a slot | License |
|---|---|---:|---:|---|---|---|
| `qwen35-27b` | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B | 262K; extensible to about 1M | Premium local generalist and final synthesizer | Current, multilingual, multimodal family with strong reasoning and agent capabilities | Apache-2.0 |
| `deepseek-r1-qwen-32b` | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 32.5B | 128K | Deliberate STEM solver and derivation checker | Long-form reasoning behavior differs from ordinary instruct tuning | MIT; Qwen-derived base |
| `devstral-small2-24b` | [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) | 24B | 256K | Code, repository analysis, debugging, and tool-oriented plans | A real software-engineering specialist gives routing a capability axis rather than only quality levels | Apache-2.0 |
| `gemma4-31b` | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | 31B | 256K | Careful generalist, visual interpretation, critique | Different model family; useful independent errors and multimodal coverage | Apache-2.0 |
| `magistral-small-24b` | [`mistralai/Magistral-Small-2509`](https://huggingface.co/mistralai/Magistral-Small-2509) | 24B | 128K | Multilingual and visual deliberate reasoner | Gives a second reasoning style and explicit, parseable thinking boundaries | Apache-2.0 |
| `phi4-reasoning-plus-14b` | [`microsoft/Phi-4-reasoning-plus`](https://huggingface.co/microsoft/Phi-4-reasoning-plus) | 14B | 32K | Compact math/reasoning critic | Tests whether a smaller specialist can cheaply catch errors from larger workers | MIT |
| `granite33-8b` | [`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | 8B | 128K | Fast decomposition, extraction, formatting, and verification | Supplies a genuinely cheap routing option and another independent family | Apache-2.0 |

The pool includes a Qwen generalist and a Qwen-derived DeepSeek checkpoint, plus
two Mistral-derived specialists. Their post-training and assigned jobs are very
different, but correlated errors remain an empirical risk. Pairwise error
overlap and the oracle-over-best-worker gap will determine whether one should be
replaced in the next iteration.

## Suggested GPU placement

The following is a conservative BF16 starting point. Weight figures are
parameter-count estimates (two bytes per parameter), not measured process RSS.

| GPU | Resident servers | Approx. raw weights | Approx. headroom before runtime overhead |
|---:|---|---:|---:|
| 0 | DeepSeek-R1-Distill-Qwen-32B + Granite-3.3-8B | 81 GB | 60 GB |
| 1 | Gemma-4-31B + Phi-4-reasoning-plus-14B | 90 GB | 51 GB |
| 2 | Qwen3.5-27B + Devstral-Small-2-24B | 102 GB | 39 GB |
| 3 | Magistral-Small-24B | 48 GB | 93 GB |
| **Total** | **7 servers** | **about 321 GB** | **about 243 GB** |

Start every server with a 16K served-context cap and bounded output tokens
(4,096 for normal workers, 8,192 for deliberate reasoners). This keeps the
first comparison focused on orchestration rather than pathological long
generations. Increase context only after recording the real KV-cache and peak
HBM numbers. Native context length describes model capability, not simultaneous
serving capacity.

The present `scripts/worker_pool.sbatch` rejects duplicate GPU assignments.
Therefore this placement is a design, not a drop-in YAML configuration. Before
using it, the launcher needs an explicit co-location mechanism and must verify
that the sum of per-process memory reservations on each GPU is below one. A
single model per GPU plus model swapping is a valid smoke-test fallback, but it
is not appropriate for latency or throughput measurements.

## Precision plan

### Primary: publisher high-fidelity checkpoints

- Best choice for the causal claim: it minimizes a quantization-quality confound.
- Fits by weights on the proposed placement.
- Establishes each worker's accuracy, complementarity, latency, and token use.
- Devstral Small 2 is published with FP8 quantization metadata; record that
  exception explicitly instead of describing the whole tier as BF16.

### Efficiency variant: 4-bit weight-only

- Prefer publisher-supplied quantizations when available. If only a community
  conversion exists, pin its revision, record its calibration method, and test
  it against the publisher checkpoint before admitting it to the tier.
- Pin exact checkpoint revisions and serving-library versions. Do not silently
  mix unrelated community quantizers into the main result.
- Validate every quantized checkpoint against its BF16 counterpart on the same
  prompts. Report accuracy delta, answer agreement, tokens/second, time to first
  token, and peak HBM.
- Accept a quantized checkpoint only if it preserves the tier's qualitative
  conclusion. A practical pre-registered guardrail is no more than 1 percentage
  point absolute loss for a worker and no more than 1 point loss in the
  conductor's gain over its equal-budget baseline. Treat these thresholds as
  experiment policy, not known properties of the checkpoints.

On H200, FP8 is also worth testing before INT4 if the serving stack supports a
well-validated conversion. H200 has native FP8 Tensor Cores; FP8 normally gives
a cleaner quality/efficiency intermediate point than jumping directly to 4-bit.

## What would prove Theo Conductor is useful?

The experiment must show more than “seven calls beat one call.” Use an untouched
test split and report all systems at matched maximum call budgets of 1, 3, 5,
and 7. At minimum compare:

1. Every worker individually.
2. Best fixed worker.
3. Random and round-robin routing.
4. Best worker repeated with self-consistency/majority vote.
5. A fixed solve-critique-synthesize workflow.
6. All workers once plus a fixed synthesizer, where the budget permits it.
7. Theo Conductor at the same call and output-token budgets.
8. Oracle routing, used only as an upper bound.

The strongest proof consists of three results together:

- **Complementarity:** oracle accuracy is materially above the best fixed
  worker, showing that useful routing opportunity exists.
- **Selection/composition value:** Theo beats fixed, random, and repeated-best
  baselines at the same maximum calls and output-token allowance.
- **Efficiency:** the gain remains when plotted against actual generated tokens,
  latency, and allocated GPU-hours rather than only nominal calls.

Pre-register the primary comparison and confidence interval. A reasonable gate
for a proof-of-concept is a statistically supported improvement over the best
equal-budget baseline, plus a positive result on at least two task families.
Do not require a particular numerical lift before seeing benchmark variance.

## Required measurements

Every result row should contain these fields so tiers remain comparable:

| Group | Fields |
|---|---|
| Identity | `tier_id`, checkpoint revisions, serving image/version, quantization method, prompt-template revision |
| Quality | accuracy/reward, 95% bootstrap CI, subject and difficulty slices, failure rate |
| Routing | model call share, mean/max calls, sequential depth, oracle gap, pairwise error overlap |
| Tokens | prompt, cached prompt, generated, and propagated intermediate tokens per workflow |
| Latency | p50/p95 end-to-end latency, time to first token, decode tokens/second |
| Compute | allocated GPU-hours, active GPU-hours if measurable, peak HBM by GPU, energy if available |
| Money | API spend or `allocated_gpu_hours * accounting_rate`, with the rate and inclusions recorded |

For local tiers, calculate:

```text
allocated GPU-hours per 1,000 workflows
  = GPU count * benchmark wall-clock hours / completed workflows * 1,000

local cost per 1,000 workflows
  = allocated GPU-hours per 1,000 workflows * internal USD/GPU-hour
```

Keep the raw GPU-hour number even when the internal price is unknown; a later
price can then be applied without rerunning the benchmark.

## Extensible tier-card template

Copy this block for every new tier. Fixed labels make Markdown comparisons easy
to generate later from YAML or JSON without changing the meaning of a field.

```yaml
tier_id: <stable-id>
status: proposed | measured | retired
hypothesis: <what this tier tests>
hardware:
  accelerator: <type>
  count: <integer>
  memory_gb_each: <number>
  allocation_scope: workers-only | workers-and-conductor
economics:
  billing_unit: usd_per_token | usd_per_gpu_hour | sunk_hardware
  input_usd_per_1m_tokens: null
  output_usd_per_1m_tokens: null
  gpu_usd_per_hour: null
serving:
  engine: <name-and-version>
  precision: <BF16|FP8|AWQ-4bit|...>
  served_context_tokens: <integer>
  default_max_output_tokens: <integer>
  residency: simultaneous | swapped
workers:
  - worker_id: <stable-id>
    checkpoint: <repository@revision>
    role: <routing role>
    gpu: <integer-or-remote>
benchmark:
  dataset_revision: <id>
  split_revision: <id>
  max_workflow_calls: [1, 3, 5, 7]
  result_artifact: null
measurements:
  accuracy: null
  ci95: null
  oracle_accuracy: null
  mean_calls: null
  generated_tokens_per_workflow: null
  workflows_per_hour: null
  allocated_gpu_hours_per_1000_workflows: null
  cost_usd_per_1000_workflows: null
```

## Risks and decision points

- **Co-location support:** the current launcher assumes exclusive GPUs and must
  be extended before all seven models can be simultaneously resident.
- **Memory fragmentation/graphs:** aggregate arithmetic is necessary but not
  sufficient. Run a seven-endpoint residency test and a concurrent worst-case
  prompt test before a long benchmark.
- **Quantized availability:** official 4-bit checkpoints are not uniform across
  the pool. BF16 is the reproducible common denominator.
- **Prompt-template differences:** DeepSeek explicitly recommends no system
  prompt, while the current OpenAI-compatible client injects a worker system
  message. Either add per-model prompt policy or measure the deviation; otherwise
  DeepSeek may be artificially weakened.
- **Multimodal path:** Gemma and Mistral provide vision capability, but the
  current benchmark/client path must actually transmit image content before this
  is counted as a demonstrated benefit.
- **Model-family correlation:** two workers are Qwen-derived and two are
  Mistral-derived. Keep them only if their distinct post-training produces
  useful complementarity in the error-overlap analysis.

## Source notes

- NVIDIA specifies 141 GB HBM3e and 4.8 TB/s bandwidth for H200:
  [H200 product page](https://www.nvidia.com/en-us/data-center/h200/).
- Parameter counts, context capabilities, licenses, and intended uses in the
  worker table come from the linked publisher model cards.
- The Devstral, Magistral, Qwen, Gemma, Phi, DeepSeek, and Granite facts above
  are grounded in their linked publisher cards; checkpoint details should still
  be pinned at benchmark time because model repositories can change.
- DeepSeek's model card recommends temperature 0.5-0.7, no system prompt, and a
  32,768-token example deployment cap. Those settings should be treated as
  model-specific policy rather than copied to the other workers.
