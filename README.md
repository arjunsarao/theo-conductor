# theo-conductor

Uses Sakana Fugu for model routing.

## Training traces

GRPO training reports normal trainer metrics to Weights & Biases by default
under the `theo-conductor` project. Each reward batch also logs a
`conductor/plans_and_worker_outputs` table containing the generated plan,
worker responses, reward, final answer, and any execution error.
For executed-workflow training, the trace also records Kimi's verdict, reason,
validated response, attempt count, and terminal judge error (if any).

The complete trace is always appended locally as JSONL, including the raw
conductor completion and parsed JSON plan:

```text
<output-dir>/traces/plans-and-worker-outputs-rank-0.jsonl
```

### Inspecting a trace locally

`trace_viewer.py` is a Streamlit viewer for these files. It shows the reward
distribution, groups the validation reasons behind the 0.0 and 0.2 reward
cohorts, and expands each record into its plan, worker outputs, final answer,
and raw conductor completion. Parsed workflows include a Graphviz DAG view.
The overview reports conductor, worker, and Kimi judge token/latency/throughput
data, plus plan structure and parallelism metrics. New traces record conductor
generation-batch latency, successful judge request performance, and actual
workflow wall time/peak concurrency; legacy traces identify unavailable
observed fields instead of estimating them.

When a selected GRPO run contains `gpu-memory.csv`, the trace viewer also
plots device memory and utilization over time. If `gpu-process-memory.csv`
(or the legacy/alternate `gpo-process-memory.csv` name) is present, it adds
per-process GPU-memory curves and a peak-memory table. The section also reports
active-window utilization, saturation and idle duty cycles, memory pressure,
cross-GPU imbalance, and a clearly labeled heuristic compute-bound assessment.

From the repository root, run:

```bash
uv run streamlit run trace_viewer.py
```

Use the sidebar's SLURM ID dropdown to select any available
`outputs/grpo-<SLURM ID>` rank-0 trace. The largest ID is selected by default
and marked as the latest run. The original `trace_viewer.html` remains
available as a dependency-free viewer when served from the repository root.

The default trace also has exact conductor-completion token counts, calculated
with the Qwen conductor tokenizer. For another trace, generate its sidecar:

```bash
./.venv/bin/python scripts/trace_token_counts.py path/to/trace.jsonl
```

The viewer marks completions at or above the configured `4096`-token generation
cap with `★`. Counts re-tokenize raw completion text and exclude special tokens.

### Querying traces from Python or a model

`theo-trace` is a JSON-first CLI for error analysis. It supports the same
reward cohorts, normalized validation failures, batches, and record drill-down
as the browser viewer, plus question-level rollout comparisons. Output is
compact JSON by default so it can be consumed directly by another model.
When a matching token-count sidecar is available, malformed completions at the
configured generation cap are classified separately as output truncations.

```bash
# Dataset overview: rewards, errors, batches, and token saturation
theo-trace summary outputs/grpo-11220/traces/plans-and-worker-outputs-rank-0.jsonl

# Failure taxonomy with representative record IDs
theo-trace errors outputs/grpo-11220/traces/plans-and-worker-outputs-rank-0.jsonl --examples 2

# Combine filters and paginate compact records
theo-trace list TRACE.jsonl --reward 0,0.2 --search "final step" --limit 20

# Fetch the complete record after discovering its ID
theo-trace show TRACE.jsonl --id 0:17

# Find questions whose rollouts disagree, ordered worst mean reward first
theo-trace questions TRACE.jsonl --min-rollouts 2 --disagreement-only --limit 30
```

Every filtering command accepts `--reward`, `--category`, `--batch`, `--rank`,
`--search`, `--question`, `--has-plan`, and `--has-error`. Use `--pretty` before
the subcommand for indented output, or invoke it without installation as
`python -m theo_conductor.trace_analysis ...`.

The reusable API is `TraceDataset.load(...)`, `TraceQuery`, and the
`summary()`, `errors()`, `query()`, `questions()`, and `get()` methods in
`theo_conductor.trace_analysis`.

Distributed runs write one file per rank. Set `--wandb-project` and
`--wandb-run-name` to name the remote run, or use `--report-to none` to keep
only the local trace.

## Format-only GRPO

Early-stage conductor training can score JSON parsing and workflow structure
without starting or calling worker-model servers:

```bash
sbatch scripts/format_only_grpo.sbatch
```

This job requests two GPUs for conductor training. The worker registry is still
used to build the prompt and validate generated `model_id` values. To execute
worker workflows during rewards, use the unified launcher:

```bash
RUN_MODE=train MODEL_CONFIG=configs/worker_pool_small.yaml \
  sbatch scripts/worker_pool.sbatch
```

The launcher supports local, remote, and mixed worker pools. It downloads,
starts, and stops only models marked `deployment.mode: local`; remote workers
are readiness-checked without being managed by the job.

The direct training CLI uses MegaScience by default; the unified Slurm launcher
defaults to HLE. Select `megascience`, `hle`, `hle-all`, `gpqa`, or the combined
`hle-gpqa` dataset with `DATASET`:

```bash
DATASET=hle-gpqa MODEL_CONFIG=configs/worker_pool_large.yaml RUN_MODE=train \
  sbatch scripts/worker_pool.sbatch
```

`DATASET_SAMPLES` optionally caps a seeded subset before splitting, and
`VALIDATION_SAMPLES` controls the held-out row count (default `200`). The
equivalent direct CLI options are `--dataset`, `--dataset-samples`, and
`--validation-samples`.

`hle` retains the established physics-adjacent HLE subset. Use `hle-all` when
the complete HLE test split is required.

## Pregenerating HLE workflows

Workflow planning can run independently from worker execution. The planning
job loads the conductor once, generates and validates each DAG, and appends one
resumable JSONL record per HLE ID. It never starts or calls worker models.

For a single GPU job over the complete HLE split:

```bash
DATASET=hle-all MODEL_CONFIG=configs/worker_pool_frontier.yaml \
  sbatch scripts/pregenerate_workflows.sbatch
```

The default output is `outputs/hle-plans-<SLURM ID>/plans.jsonl`, accompanied
by `manifest.json`, `invalid.jsonl`, and the conductor server log. Re-submitting
with the same `PLAN_OUTPUT_DIR` skips IDs already recorded. Set
`PLAN_RETRY_INVALID=1` to retry only failed records.

For ten shards with at most four conductor jobs active at once:

```bash
PLAN_JOB_ID=$(sbatch --parsable --array=0-9%4 \
  --export=ALL,DATASET=hle-all,PLAN_SHARDS=10 \
  scripts/pregenerate_workflows.sbatch)

sbatch --dependency="afterok:${PLAN_JOB_ID}" \
  --export=ALL,PLAN_JOB_ID="${PLAN_JOB_ID}",PLAN_SHARDS=10 \
  scripts/merge_workflow_shards.sbatch
```

The dependent CPU job verifies the shard count and produces the canonical
`plans.jsonl`. Inspect it without loading any models:

```bash
uv run theo-plan summary outputs/hle-plans-<SLURM ID>
uv run theo-plan list outputs/hle-plans-<SLURM ID> --invalid-only
uv run theo-plan show outputs/hle-plans-<SLURM ID> --id hle-<ID>
```

The existing viewer discovers merged `outputs/hle-plans-*` runs and renders
their workflow DAGs alongside GRPO traces:

```bash
uv run streamlit run trace_viewer.py
```

Useful planning overrides include `DATASET_SAMPLES`, `PLAN_CONCURRENCY`,
`PLAN_MAX_TOKENS`, `PLAN_ATTEMPTS`, `PLAN_TEMPERATURE`,
`CONDUCTOR_SOURCE_MODEL`, `CONDUCTOR_LORA_PATH`, and `PLAN_OUTPUT_DIR`.
`CONDUCTOR_SOURCE_MODEL` selects the base checkpoint. When
`CONDUCTOR_LORA_PATH` is set, vLLM serves that adapter under
`CONDUCTOR_MODEL` (default `theo-conductor`) and records its path in the
manifest. Model and dataset downloads use `~/.cache/huggingface/` by default;
set `THEO_HF_HOME` to override it.

## End-to-end HLE workflow benchmark

Pregenerated plans can be executed, judged, and summarized as a resumable
dataset benchmark. Each result records the complete plan and worker outputs,
the extracted final answer, token usage, configured-price cost estimate,
workflow latency and peak concurrency, judge verdict, and any item-level
error. A hash of the plan, worker configuration, and execution settings keeps
changed runs separate even when they share an output file.

Start with ten workflows using one of the existing complete HLE plan runs:

```bash
OPENROUTER_API_KEY=... uv run theo-workflow-benchmark \
  --plans outputs/hle-plans-20484 \
  --config configs/worker_pool_frontier.yaml \
  --output-dir outputs/hle-workflow-smoke-10 \
  --max-samples 10 \
  --concurrency 2
```

The default Kimi judge and GLM fallback use `KIMI_API_KEY` and `GLM_API_KEY`.
Pass `--no-judge` to validate workflow execution before configuring those
services. Rerunning the same command resumes completed workflows and judge
verdicts. Increase `--max-samples` to `50`, then `500`. Subsequent 500-item
shards can use `--offset 500`, `--offset 1000`, and so on; use a distinct
output directory for each official scored configuration. After fixing a
transient endpoint or credential failure, add `--retry-failures` to replace
the latest failed records without repeating successful workflows.

The default uses each model's `max_output_tokens` from the YAML configuration.
The frontier budgets are grounded in observed maxima from an artificial HLE
analysis benchmark, with about 20% headroom and upward rounding to 4,096-token
boundaries: Gemini 12,288; GPT 16,384; Grok 24,576; Kimi 32,768; Claude 40,960;
DeepSeek 40,960; and GLM 65,536. The GPT and Claude observations are
closest-family proxies because the benchmark versions differ from the models
in the frontier pool. Use `--max-worker-tokens N` for one fixed pool-wide cap.

For runs where truncation is less acceptable than potentially extreme cost,
use `--use-model-context-limit`. Every worker request will then use that
model's `context_length`. This is the advertised total context limit, not a
provider-guaranteed output limit: providers may enforce a smaller output cap
or reject a request when the input plus requested output exceeds the actual
context window.

The trainable conductor comes from the selected YAML file's top-level
`conductor_model` field (`Qwen/Qwen2.5-7B` for the small-local config and
`Qwen/Qwen3.8-27B` for the large-local and frontier configs). Training updates LoRA adapters
over all linear layers; use `--lora-rank`, `--lora-alpha`, and
`--lora-dropout` to tune the adapter, or `--model-name` to override the
configured base model.

Executed-workflow training uses Kimi K2.6 as the primary semantic correctness
judge and GLM 5.2 as its fallback. Every valid rollout is sent as its own judge
request, with up to 16 requests in flight by default. Both judges use a strict
JSON Schema structured-output response; malformed or invalid workflows retain
their structural reward without being answer-judged. API or schema-validation
failures are isolated to the affected item and then sent to GLM. A response that
already hit a backend's output ceiling moves directly to the fallback rather
than repeating the same request. If both judges fail, that rollout keeps its
neutral valid-workflow reward and records the error while other verdicts and
training continue. The end-to-end preflight still requires a successful remote
verdict, so a broken judge configuration fails before the full run starts.
Configure Kimi with `KIMI_BASE_URL`, `KIMI_API_KEY`, and `KIMI_MODEL`, and GLM
with `GLM_BASE_URL`, `GLM_API_KEY`, and `GLM_MODEL`; tune failure
handling with `--judge-attempts`, `--judge-retry-delay-seconds`,
`--judge-max-tokens`, `--judge-concurrency`, `--judge-connect-timeout-seconds`,
`--fallback-judge-max-tokens`, and `--judge-timeout-seconds`. The default Kimi
budget is 16,384 output tokens and the default GLM fallback budget is 32,768
output tokens. These budgets include the model's reasoning tokens as well as
its JSON verdict. Judge clients disable
the OpenAI SDK's internal retries; training traces record the total Kimi and GLM
attempts used. Remote worker workflows execute with up to 16 rollouts in flight
by default. Each worker generation uses that model's configured
`context_length`; use `--max-worker-tokens` to impose a smaller pool-wide
ceiling. Tune concurrency with `--workflow-concurrency`.

## Small-model MegaScience benchmark

Benchmark every model in `configs/worker_pool_small.yaml` with one independent
call on the same deterministic 200-row MegaScience validation subset used by
training:

```bash
RUN_MODE=benchmark MODEL_CONFIG=configs/worker_pool_small.yaml \
  sbatch scripts/worker_pool.sbatch
```

The job starts all three vLLM endpoints, verifies them, and writes resumable
per-question records to `outputs/megascience-worker-pool/results.jsonl` and
aggregate metrics to `outputs/megascience-worker-pool/summary.json`. Metrics
include accuracy with a bootstrap 95% confidence interval, accuracy by subject,
token usage, latency, request failures, and missing-`FINAL:` extraction failures.
Re-running the command resumes completed model/question pairs.

Kimi K2.6 judges semantic correctness after generation by default, with GLM 5.2
used after Kimi exhausts its attempts and multiple answers packed into each API
request. Both use the same strict JSON Schema structured-output contract. Each
JSONL record adds `judge_correct`,
`judge_reason`, `judge_response`, `judge_model`, and `judge_error`; the top-level
`correct` field contains the authoritative judge verdict. Judge progress is
atomically checkpointed and resumes on rerun. Set
`KIMI_BASE_URL`, `KIMI_API_KEY`, or `KIMI_MODEL` to override the primary and
`GLM_BASE_URL`, `GLM_API_KEY`, or `GLM_MODEL` to override the fallback. Use
`--judge-batch-size` and `--judge-concurrency` to tune judge throughput, or pass
`--no-judge` to disable judging.

To judge or re-judge an existing results file and refresh its `summary.json`:

```bash
uv run python scripts/judge_megascience_results.py
# Add --force to replace successful verdicts already written by the same judge.
```

For an endpoint setup that is already running, invoke the benchmark directly:

```bash
uv run theo-benchmark
```

Use `--max-samples 5` for a smoke run. Dataset identity is controlled by
`--dataset`, `--seed`, `--total-samples`, and `--validation-samples`.

## Worker-pool launcher

`scripts/worker_pool.sbatch` is the single SLURM entry point for configured
worker pools:

```bash
# Start local workers, verify all local/remote endpoints, then exit.
RUN_MODE=smoke sbatch scripts/worker_pool.sbatch

# Benchmark a mixed large-model pool.
RUN_MODE=benchmark DATASET=hle-gpqa \
  MODEL_CONFIG=configs/worker_pool_large.yaml \
  sbatch scripts/worker_pool.sbatch

# Train with executed worker workflows.
RUN_MODE=train MODEL_CONFIG=configs/worker_pool_large.yaml \
  sbatch --gres=gpu:5 scripts/worker_pool.sbatch
```

The large pool runs DeepSeek V4 Flash with tensor parallelism across GPUs
`0–3` and places the 27B LoRA conductor on GPU `4`, so training requests five
H200s in total. A worker-only smoke test or benchmark needs four:

```bash
RUN_MODE=smoke MODEL_CONFIG=configs/worker_pool_large.yaml \
  sbatch --gres=gpu:4 scripts/worker_pool.sbatch

RUN_MODE=benchmark MODEL_CONFIG=configs/worker_pool_large.yaml \
  sbatch --gres=gpu:4 scripts/worker_pool.sbatch
```

The default `#SBATCH` allocation remains eight GPUs because the small pool
still uses six worker GPUs plus two training GPUs; the command-line `--gres`
override selects the smaller large-pool allocation.

```bash
RUN_MODE=train MODEL_CONFIG=configs/worker_pool_small.yaml \
  sbatch scripts/worker_pool.sbatch
```

Local deployment settings live beside each model's client configuration:
`source_model`, `gpu_set`, `tensor_parallel_size`, `max_model_len`, and
optional `kv_cache_dtype` and `gpu_memory_utilization`. Remote entries need only
`deployment.mode: remote`. Each config's `train_gpu_set` selects conductor
training devices and can be overridden with `TRAIN_GPUS`; `VLLM_EXTRA_ARGS`
appends flags to every locally managed worker server. In training mode the
launcher uses GPU 0 for PyTorch/LoRA training and starts a dedicated TRL vLLM
generation server on GPU 1. Generation placement and capacity can be tuned with
`GENERATION_GPU`, `GENERATION_VLLM_GPU_MEMORY_UTILIZATION`, and
`GENERATION_VLLM_MAX_MODEL_LEN`.

Worker entries may declare `cost_per_1m_input_tokens` and
`cost_per_1m_output_tokens` in USD. When an endpoint reports token usage, traces
include an estimated request cost and the trace viewer reports mean and total
cost by model. These are configured list-price estimates and do not account for
provider routing, cache discounts, or tiered pricing.

Benchmark mode uses the same `DATASET` and `VALIDATION_SAMPLES` settings.
`BENCHMARK_TOTAL_SAMPLES` and `BENCHMARK_VALIDATION_SAMPLES` provide
benchmark-specific overrides. Its default output directory is
`outputs/<dataset>-worker-pool`.

# TODO

- Add bigger models (120B class models)
- Add term to penalize overly sequential workflows.

Compare the breakdown
