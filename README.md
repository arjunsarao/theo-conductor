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

The viewer marks completions at or above the configured `1024`-token generation
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

Training and its held-out evaluation split use MegaScience by default. Select
`megascience`, `hle`, `gpqa`, or the combined `hle-gpqa` dataset with
`DATASET`:

```bash
DATASET=hle-gpqa MODEL_CONFIG=configs/worker_pool_large.yaml RUN_MODE=train \
  sbatch scripts/worker_pool.sbatch
```

`DATASET_SAMPLES` optionally caps a seeded subset before splitting, and
`VALIDATION_SAMPLES` controls the held-out row count (default `200`). The
equivalent direct CLI options are `--dataset`, `--dataset-samples`, and
`--validation-samples`.

The trainable conductor comes from the selected YAML file's top-level
`conductor_model` field (`Qwen/Qwen2.5-7B` for the small-local config and
`Qwen/Qwen3.5-27B` for the large-local config). Training updates LoRA adapters
over all linear layers; use `--lora-rank`, `--lora-alpha`, and
`--lora-dropout` to tune the adapter, or `--model-name` to override the
configured base model.

Executed-workflow training uses Kimi K2.6 as the sole semantic correctness
judge. Every valid rollout is sent as its own judge request, with up to 256
requests in flight by default; malformed or invalid workflows retain their
structural reward without being answer-judged. API or schema-validation
failures retry only the affected item, and training stops if all attempts for
an item fail—there is no local exact/numeric heuristic fallback. Configure the
endpoint with `KIMI_BASE_URL`, `KIMI_API_KEY`, and `KIMI_MODEL`; tune failure
handling with `--judge-attempts`, `--judge-retry-delay-seconds`,
`--judge-max-tokens`, `--judge-concurrency`, `--judge-connect-timeout-seconds`,
and `--judge-timeout-seconds`. The default 8,192-token judge budget includes
Kimi's reasoning tokens as well as its JSON verdict. Judge clients disable the
OpenAI SDK's internal retries so `--judge-attempts` is the exact number of item
attempts recorded in training traces.

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

Kimi K2.6 judges semantic correctness after generation by default, with multiple
answers packed into each API request. Each JSONL record adds `judge_correct`,
`judge_reason`, `judge_response`, `judge_model`, and `judge_error`; the top-level
`correct` field contains the authoritative judge verdict. Judge progress is
atomically checkpointed and resumes on rerun. Set
`KIMI_BASE_URL`, `KIMI_API_KEY`, or `KIMI_MODEL` to override the cluster
defaults. Use `--judge-batch-size` and `--judge-concurrency` to tune judge
throughput, or pass `--no-judge` to disable judging.

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
  sbatch scripts/worker_pool.sbatch
```

Local deployment settings live beside each model's client configuration:
`source_model`, `gpu_set`, `tensor_parallel_size`, `max_model_len`, and
optional `gpu_memory_utilization`. Remote entries need only
`deployment.mode: remote`. `TRAIN_GPUS` selects conductor devices (default
`6,7`), while `VLLM_EXTRA_ARGS` appends flags to every locally managed vLLM
server.

Benchmark mode uses the same `DATASET` and `VALIDATION_SAMPLES` settings.
`BENCHMARK_TOTAL_SAMPLES` and `BENCHMARK_VALIDATION_SAMPLES` provide
benchmark-specific overrides. Its default output directory is
`outputs/<dataset>-worker-pool`.

# TODO

- Add bigger models (120B class models)
- Add term to penalize overly sequential workflows.

Compare the breakdown
