# theo-conductor

Uses Sakana Fugu for model routing.

## Resumable evaluation

`theo-evaluate` is the shared entry point for new evaluation runs. The module
form below also works before refreshing the installed console entry points.
It uses standard completion requests, including for clients configured with
native batch support; the existing provider batch launchers remain available.
All model arguments are **registry IDs** (`model_idx` in the YAML), rather than
provider model slugs. Use a configuration containing the models needed for
each stage. Plan generation checks that the prompt's worker catalogue exists
in that configuration.

Input is JSONL with one question per line:

```json
{"id":"question-1","question":"What is 2 + 2?","answer":"4","subject":"Math"}
```

Optional fields include `reference_answer`, `answer_type`, and `image` (a URL
or embedded image data URI). Input IDs must be unique. Each independent sample
gets its own `sample_index`; `--attempts` controls recovery attempts within that
sample and never increases k.

```bash
# Generate plans only, with four independent samples per question.
.venv/bin/python -m theo_conductor.evaluate plans \
  --input questions.jsonl --config configs/worker_pool_frontier.yaml \
  --planner-model gpt-6.1-sol --samples 4 --output-dir outputs/plans-run

# Execute those stored plans without making new plans.
.venv/bin/python -m theo_conductor.evaluate workflows \
  --plans outputs/plans-run --config configs/worker_pool_frontier.yaml \
  --judge-model gemini-3.7-flash --output-dir outputs/workflows-run

# Generate plans and execute workflows in one resumable run.
.venv/bin/python -m theo_conductor.evaluate workflows \
  --input questions.jsonl --config configs/worker_pool_frontier.yaml \
  --planner-model gpt-6.1-sol --samples 4 \
  --judge-model gemini-3.7-flash --output-dir outputs/full-run

# Benchmark exactly one model; omit --judge-model to defer judging.
.venv/bin/python -m theo_conductor.evaluate model \
  --input questions.jsonl --config configs/worker_pool_frontier.yaml \
  --model gpt-6.1-sol --samples 4 --output-dir outputs/direct-run

# Judge existing completions in a managed run; generation is never invoked.
.venv/bin/python -m theo_conductor.evaluate judge \
  --output-dir outputs/direct-run --config configs/worker_pool_frontier.yaml \
  --judge-model gemini-3.7-flash

# Import and judge a legacy, single-model results file in a new directory.
.venv/bin/python -m theo_conductor.evaluate judge \
  --results outputs/old-run/results.jsonl --output-dir outputs/rejudged-run \
  --config configs/worker_pool_frontier.yaml --judge-model gemini-3.7-flash

# Catalogue managed runs and existing legacy summaries.
.venv/bin/python -m theo_conductor.evaluate catalog --root outputs

# Aggregate independent runs with the same inputs, model and generation settings.
.venv/bin/python -m theo_conductor.evaluate aggregate \
  --runs outputs/direct-run outputs/direct-run-2 --output reports/aggregate.json
```

To resume, repeat the same command with the same output directory. Failed and
interrupted items retry automatically. Completed plans, solver responses, and
judge verdicts are reused when their request fingerprints still match. Worker
steps checkpoint separately: a failed layer retains its successful siblings,
and retrying a failed or capped step does not rerun its successful ancestors.
Steps ending with `length`/`max_tokens`, empty responses, provider errors, and
content filtering are treated as incomplete; their dependents wait until the
step succeeds. Raise `--max-tokens` to recover a persistent cap, or change the
model's output limit in the configuration. Changed requests invalidate affected
steps and their descendants. Plans reuse only when the planner, prompt, inputs,
and generation parameters match.

For a recovery that must leave completed workflow slots untouched even if their
cache fingerprints differ, add `--retry-failed` to the original `workflows`
command. It skips completed slots
before planning, execution and judging. Run `judge` separately for any completed
answers that still need grading.

Use `--request-timeout-seconds` for a hard elapsed deadline per model request
(including hosted shell execution), and `--workflow-timeout-seconds` for a hard
deadline per item across planning, retries and judging, excluding queue time.
For example, `--request-timeout-seconds 1800 --workflow-timeout-seconds 7200`
bounds each request to 30 minutes and each item to two hours. These limits are
optional; successful checkpoints survive timeouts. Usage unavailable after a
timeout remains unknown, and cancelling a local request does not guarantee
that provider execution or billing stops immediately.

You can increase `--samples` in an existing question-input run to add independent
samples without repeating completed samples. Execution from `--plans` preserves
the samples already present in the source file. Use a new output directory for
a different input, mode, solver, or planner. `--force` replaces judge verdicts;
it does not force generation. Concurrent writers to the same run are rejected.

Each managed run contains:

- `manifest.json`: input/model identity, sample count, and configuration revision hashes.
- `checkpoints/*.json`: durable per-question/sample state, worker outputs, and attempt history.
- `results.jsonl`: sorted snapshot of all samples, including pending and failed samples.
- `plans.jsonl`: generated/imported plans, usable by the workflow command.
- `summary.json`: shared statistics and repeated-sampling metrics.
- `failures.jsonl`: current generation/judge failures, failed steps, and capped step IDs.

The summary reports planned and executed model distributions; planner, worker,
solver, tool, and judge token/cost/latency statistics; historical failure reasons;
and current generation/judge failure counts. Costs distinguish provider-reported
charges from configured list-price estimates. Every metric includes known and
unknown observation counts; missing usage is not treated as zero. Attempt totals
include recovery attempts, and each worker tool round checkpoints its usage.
Stored-plan and imported-completion statistics are reported separately from
requests issued by the current run, with an additional observed end-to-end total.
Legacy imports retain their completions and metadata; historical request costs
cannot be reconstructed when they were not recorded.

For n independent samples per question, `mean_at_k` is the mean correctness of
samples (equivalently the expected mean of a random k-subset). `pass_at_k` uses
`1 - C(n-c, k) / C(n, k)`, where c is the number of correct samples. The summary
reports every k from 1 through n; at k=n, pass@k is the fraction of questions with
at least one correct sample. Recovery retries do not count as samples. Metrics
remain null until every question has all expected outcomes. Exhausted solver or
workflow failures count as unsuccessful samples; missing plans and judge errors
remain unknown. Plan-only runs have no correctness metrics.

Snapshots use fsync plus atomic replacement and are refreshed on a graceful
interrupt. After a hard kill, per-item checkpoints remain authoritative and the
next invocation rebuilds exported snapshots. A completion request interrupted
before its response is saved may be sent again; these APIs cannot guarantee
exactly-once billing. An interrupted tool step restarts that step, so tools with
external side effects need their own idempotency support. Existing provider
batch launchers retain their separate batch lifecycle behavior.

For SLURM, create `slurm_logs` before submission and pass the same CLI arguments:

```bash
mkdir -p slurm_logs
sbatch scripts/evaluate.sbatch workflows \
  --input questions.jsonl --config configs/worker_pool_frontier.yaml \
  --planner-model gpt-6.1-sol --samples 4 --output-dir outputs/full-run
```

## PRLBench comparison

Benchmark Fugu Ultra v2 and the GPT-6.1 Sol conductor on all 100 PRLBench
paper-level task bundles, with **four independent attempts per paper per system**,
Gemini 3.8 Flash criterion judging, and the same OpenRouter hosted shell with
networking disabled:

```bash
.venv/bin/python -m theo_conductor.prlbench_comparison prepare \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter
.venv/bin/python -m theo_conductor.prlbench_comparison run \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter
```

The report includes mean@4, pass@4, raw criterion scores and subfield metrics.
An attempt passes at 70% criterion satisfaction. Public criterion weights are
unavailable, so the raw score uses a disclosed equal-weight adaptation.
See [the PRLBench guide](docs/prlbench.md) for interpreter isolation and limits,
Slurm launching, provenance, resume and judge-only recovery.

## Radar comparisons

`theo-radar` produces one conductor-versus-system SVG per comparison in
`figures/`, with side-by-side mean@k and pass@k panels, a shared 0–100% scale,
category question counts, and overall scores. SVGs open in a browser and stay
sharp when exported. No model calls or plotting dependencies are needed.
Fugu comparisons use `#e10600`, OpenAI/GPT comparisons use `#0fa37f`, and
Jev-Router uses `#e752bb`. Claude comparisons retain `#ca603b`; the conductor
uses `#282d36`.
Use the module form before refreshing installed console entry points:

```bash
.venv/bin/python -m theo_conductor.radar \
  --dataset hle --dataset-label HLE-Diamond-Physics \
  --metadata outputs/hle-diamond-physics-classifications-gpt-5.6-sol/classifications.jsonl \
  --system Conductor outputs/hle-diamond-gpt61-frontier-mean4-22782 \
  --system "Fable 5.1" outputs/hle-diamond-fable51-anthropic-batch-mean4

.venv/bin/python -m theo_conductor.radar \
  --dataset gpqa-d --dataset-label GPQA-Diamond-Physics \
  --metadata path/to/gpqa_diamond.csv \
  --system Conductor path/to/comparison/conductor \
  --system "Fugu Ultra v2" path/to/comparison/fugu
```

Repeat `--system LABEL PATH [PATH ...]` to add systems or supply multiple
independent result files. A directory resolves to `results.jsonl`, or to its
`run-*/results.jsonl` files when there is no top-level snapshot. Managed
`sample_index` slots and legacy single-sample files are supported; duplicate
slots are rejected rather than counted as retries. Only combine files from the
same system configuration and independent runs; the command cannot establish
historical configuration compatibility from legacy records alone.

HLE axes default to `tag_1`; `--tag tag_2`, `--tag tag_3`, or `--tag all`
select other classifications. With `all`, category groups overlap. GPQA-D axes
use the CSV's `Subdomain`, joined by `Record ID` to `gpqa-diamond-<Record ID>`.
Alternatively supply JSONL with `id` (or `example_id`) and `subdomain`.
`--metadata-format csv` supports cached CSV files without a `.csv` extension.

The default k is the independent sample count; `--k` selects a smaller k using
the shared combinatorial pass@k estimator. All systems must have identical
question IDs, equal sample counts, complete outcomes, and metadata for every
question; incomplete cohorts are rejected without dropping questions.
Exhausted generation failures count as unsuccessful samples; missing plans,
judge errors, and unknown verdicts block plotting. At least three nonempty
categories are required for a radar chart. Small category counts can be
unstable, and pass@k is an oracle metric, not answer-selection accuracy.

`--output-dir` changes the destination and `--prefix` changes the filename
prefix. `--dataset-label` sets the displayed cohort name; use the full name of
the evaluated subset, such as `HLE-Diamond-Physics` or `GPQA-Diamond-Physics`.
The accompanying `<prefix>-radar-metrics.json` records source paths and
hashes, paired question IDs, denominators, per-category/overall metrics,
generation failures, and comparison limitations. These are comparisons of the
saved configured systems; equal k does not establish matched compute budgets.

## FrontierScience track comparisons

`theo-bars` creates a grouped-bar comparison of Conductor and Fugu Ultra v2 on
FrontierScience-physics, with Research and Olympiad groups and mean@k/pass@k
bars. It reads the saved results and track metadata, uses the same paired-cohort
and failure policies as the radar charts, and retains original source rows,
including repeated question texts. The default k is all independent samples.

```bash
.venv/bin/python -m theo_conductor.grouped_bars \
  --comparison-root outputs/frontierscience-physics-fugu-vs-conductor-m4-20261003 \
  --png
```

The chart and its JSON metrics/provenance file are saved in `figures/` by
default; `--output-dir` changes the destination. SVG generation needs no extra
dependencies. Optional `--png` also exports a PNG using installed ImageMagick
(`convert`). Research correctness uses the run's recorded rubric verdicts;
Olympiad uses its recorded short-answer verdicts. No new model calls are made.

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

This job requests two GPUs for conductor training. The complete conductor
prompt, including model guidance and examples, lives in
`conductor-prompt-refined.md`. Training, planning, and generation scripts use
this file and substitute only `<USER QUESTION>`. Keep its worker catalogue in
sync with the worker pool used for a run. The worker registry supplies execution
settings and validates generated `model_id` values. To execute
worker workflows during rewards, use the unified launcher:

```bash
RUN_MODE=train MODEL_CONFIG=configs/worker_pool_small.yaml \
  sbatch scripts/worker_pool.sbatch
```

The launcher supports local, remote, and mixed worker pools. It downloads,
starts, and stops only models marked `deployment.mode: local`; remote workers
are readiness-checked without being managed by the job.

The direct training CLI uses MegaScience by default; the unified Slurm launcher
defaults to HLE. Select `megascience`, `hle`, `hle-text`, `hle-all`, `gpqa`, or the combined
`hle-gpqa` dataset with `DATASET`:

```bash
DATASET=hle-gpqa MODEL_CONFIG=configs/worker_pool_large.yaml RUN_MODE=train \
  sbatch scripts/worker_pool.sbatch
```

`DATASET_SAMPLES` optionally caps a seeded subset before splitting, and
`VALIDATION_SAMPLES` controls the held-out row count (default `200`). The
equivalent direct CLI options are `--dataset`, `--dataset-samples`, and
`--validation-samples`.

`hle` retains the established physics-adjacent HLE subset.
`hle-physics-text` is the 202-question subset whose category is exactly
`Physics` and whose HLE `image` field is empty. `hle-text` contains the 2,158
canonical text-only questions across every category. Use `hle-all` only when
the complete 2,500-question split and proper multimodal input support are
required.

## Pregenerating HLE workflows

Workflow planning can run independently from worker execution. The planning
job loads the conductor once, generates and validates each DAG, and appends one
resumable JSONL record per HLE ID. It never starts or calls worker models.

For a single GPU job over the text-only HLE split:

```bash
DATASET=hle-text MODEL_CONFIG=configs/worker_pool_frontier.yaml \
  sbatch scripts/pregenerate_workflows.sbatch
```

The default output is `outputs/hle-plans-<SLURM ID>/plans.jsonl`, accompanied
by `manifest.json`, `invalid.jsonl`, and the conductor server log. Re-submitting
with the same `PLAN_OUTPUT_DIR` skips IDs already recorded. Set
`PLAN_RETRY_INVALID=1` to retry only failed records.

For ten shards with at most four conductor jobs active at once:

```bash
PLAN_JOB_ID=$(sbatch --parsable --array=0-9%4 \
  --export=ALL,DATASET=hle-text,PLAN_SHARDS=10 \
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

Its **HLE · workflow benchmark** page also discovers benchmark directories
containing `summary.json` and `results.jsonl`. It reports overall and
completed-only accuracy, external-judge coverage, failures, costs, latency,
capped outputs, worker-model usage, workflow DAGs, and question-level details.

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

Workflow benchmarking is text-only by default, including when the supplied
legacy plan file was generated from `hle-all`: canonical HLE IDs are joined
back to the dataset, image-dependent plans are removed before `--offset` and
`--max-samples` are applied, and subject metadata is restored. Pass
`--include-multimodal` only after worker requests actually carry HLE images.

Start with ten workflows using one of the existing complete HLE plan runs:

```bash
OPENROUTER_API_KEY=... uv run theo-workflow-benchmark \
  --plans outputs/hle-plans-20484 \
  --config configs/worker_pool_frontier.yaml \
  --output-dir outputs/hle-workflow-smoke-10 \
  --max-samples 10 \
  --concurrency 2
```

Workflow benchmarking uses OpenRouter's `~deepseek/deepseek-flash-latest` by
default and judges each workflow before its completed JSONL checkpoint is
written; it therefore only requires `OPENROUTER_API_KEY`. Pass `--no-judge` to
validate workflow execution before configuring the judge, or
`--no-judge-immediately` to retain the older after-run batched judging mode.
Rerunning the same command resumes completed workflows and judge verdicts.
Increase `--max-samples` to `50`, then `500`. Subsequent 500-item
shards can use `--offset 500`, `--offset 1000`, and so on; use a distinct
output directory for each official scored configuration. After fixing a
transient endpoint or credential failure, add `--retry-failures` to replace
the latest failed records without repeating successful workflows.

To run a fresh GPT-6 Astra conductor benchmark on the canonical 202-question
HLE Physics text subset, submit the dedicated Slurm job:

```bash
sbatch scripts/benchmark_gpt6_frontier_hle_physics.sbatch
```

It first has GPT-6 Astra generate all 202 workflow plans, then uses
`worker_pool_frontier.yaml` to execute them and checkpoints a DeepSeek Flash
verdict as soon as each workflow completes.

### Dependency-layer worker batching

`--concurrency` runs ordinary worker requests concurrently; it does not turn
them into provider batch jobs. To pool all ready steps across workflows by DAG
layer and worker model, pass `--batch-workers`:

```bash
OPENROUTER_API_KEY=... uv run theo-workflow-benchmark \
  --plans outputs/hle-plans-20484 \
  --config configs/worker_pool_frontier.yaml \
  --output-dir outputs/hle-batched \
  --batch-workers \
  --worker-batch-size 50000
```

The frontier configuration enables `native_batch` for GPT, Claude, Gemini,
and Kimi. Each non-empty batch-enabled model bucket is submitted inline to
`POST /api/beta/batches`, then polled through `GET /api/beta/batches/:id` until
it completes. OpenRouter requires one model per batch, so the scheduler's
model buckets map directly to its API. The Batch Beta is text-only and uses a
24-hour completion window; leave `--include-multimodal` off in this mode.

Direct OpenAI-compatible clients can also set `native_batch: true`. Non-
OpenRouter endpoints use the standard Files and `/v1/batches` protocol. A
direct OpenAI worker reads its credential from `OPENAI_API_KEY`; OpenRouter
workers use `OPENROUTER_API_KEY` as before.

Models without native batch support still participate in the same global DAG
schedule, but their bucket is executed as ordinary requests bounded by
`--concurrency`. Thus the mode is safe for mixed worker pools. Batch jobs are
asynchronous and dependency layers are sequential, so this mode targets fewer
requests and provider batch pricing rather than minimum wall-clock latency.

Run a CPU-only 500-question shard on the `train` Slurm partition instead of a
login node with:

```bash
sbatch scripts/workflow_benchmark.sbatch
```

The default shard is offset 0. Override it explicitly with
`BENCHMARK_OFFSET=500`, or submit all five text-only HLE shards as an array;
the final task naturally processes only the remaining questions:

```bash
BENCHMARK_OFFSET=500 sbatch scripts/workflow_benchmark.sbatch
sbatch --array=0-4 scripts/workflow_benchmark.sbatch
```

`BENCHMARK_SHARD_SIZE`, `BENCHMARK_BASE_OFFSET`, `BENCHMARK_OUTPUT_DIR`,
`BENCHMARK_CONCURRENCY`, and `WORKER_BATCH_SIZE` provide the corresponding
overrides. Set `BENCHMARK_RETRY_FAILURES=1` to retry checkpointed errors or
`BENCHMARK_JUDGE=0` for a worker-only run. Slurm command-line options can
override the script allocation, for example `sbatch --partition=cpu ...` on a
cluster with a dedicated CPU partition. No GPU resource is requested.

The default uses each model's `max_output_tokens` from the YAML configuration.
The frontier pool uses OpenRouter's advertised maximum completion limits for
DeepSeek (393,216), Kimi (943,718), and GLM (131,072), checked on October 2, 2026.
The remaining caps are benchmark-based budgets: Gemini 12,288; GPT 32,768;
Grok 24,576; and Claude 65,536. These budgets were based on observed output
lengths with headroom and pilot adjustments, rather than provider maximums.
The GPT and Claude original observations used closest-family proxies.
Use `--max-worker-tokens N` for one fixed pool-wide cap.

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

## FrontierScience Physics benchmark

OpenAI's [FrontierScience dataset](https://huggingface.co/datasets/openai/frontierscience)
is available as `frontierscience-physics` (70 public test questions tagged
exactly `physics`), `frontierscience-olympiad-physics` (50), and
`frontierscience-research-physics` (20). These selectors are evaluation-only
and excluded from training choices. IDs include the track, source
`task_group_id`, and the original track row index;
problem text and gold answers are preserved verbatim.
The public Research physics file contains an exact duplicate: the 20 rows
represent 19 distinct questions. The loader retains all source rows; disclose
this duplication when reporting scores and do not count it as independent evidence.

Export the normalized test set for the resumable evaluation commands:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from theo_conductor.data import load_conductor_dataset

Path("outputs/frontierscience").mkdir(parents=True, exist_ok=True)
load_conductor_dataset("frontierscience-physics").to_json(
    "outputs/frontierscience/physics.jsonl"
)
PY
.venv/bin/python -m theo_conductor.evaluate model \
  --input outputs/frontierscience/physics.jsonl \
  --config configs/worker_pool_frontier.yaml --model gpt-6.1-sol \
  --output-dir outputs/frontierscience/direct
```

The same JSONL works with `plans` and `workflows`. To use `theo-benchmark`,
pass `--dataset frontierscience-physics --full-dataset`; train/test splitting
is rejected for this benchmark. Loading requires access to Hugging Face.

Research rows select rubric grading automatically: the judge assesses the full
response against the supplied ten-point rubric, and `judge_score >= 7` determines
correctness. Numeric scores are retained in `judge_score`. Olympiad rows use
short-answer semantic grading. The judge model and number of independent samples
remain configurable; these settings must be disclosed when comparing with OpenAI.
Gold answers and rubrics are grading data and are not sent to solvers or planners.

## Frontier HLE Physics benchmark

Benchmark every frontier worker independently on all 202 text-only HLE Physics
questions with a CPU-only Slurm job:

```bash
OPENROUTER_API_KEY=... sbatch scripts/benchmark_frontier_hle_physics.sbatch
```

GPT and Claude are submitted through the provider Batch API; the other
frontier models use bounded concurrent requests. The job fails before making
model calls if the canonical filtered dataset is not exactly 202 rows, uses
each model's configured output limit, and checkpoints resumable results under
`outputs/hle-physics-frontier/`. Semantic judging is enabled by default using
the configured Kimi judge and GLM fallback. Set `BENCHMARK_JUDGE=0` for
generation only, or set `BENCHMARK_OUTPUT_DIR` to keep separate runs.

To benchmark only GPT-5.5 and Claude Opus 4.8 with ordinary concurrent
requests, run:

```bash
OPENROUTER_API_KEY=... sbatch scripts/benchmark_premium_hle_physics.sbatch
```

This stores resumable results under `outputs/hle-physics-premium-live/`. Set
`BENCHMARK_CONCURRENCY` to tune concurrent requests per model, or set
`BENCHMARK_JUDGE=0` to skip semantic judging.

Evaluate the Qwen3.8-27B conductor end to end on the same question set with:

```bash
OPENROUTER_API_KEY=... scripts/evaluate_qwen38_conductor_hle_physics.sh
```

The launcher submits a one-GPU planning job and a dependent CPU-only workflow
job, releasing the GPU before remote worker inference begins. By default it
evaluates the base `Qwen/Qwen3.8-27B` conductor. Set `CONDUCTOR_LORA_PATH` to
evaluate a trained adapter instead. Plans are stored under
`outputs/hle-plans-<planning-job-id>/`, and scored workflow results under
`outputs/hle-physics-qwen38-conductor-<planning-job-id>/`.

To benchmark Qwen3.8-27B itself as a direct solver instead—without generating
plans or calling frontier workers—run:

```bash
sbatch scripts/benchmark_qwen38_direct_hle_physics.sbatch
```

This job starts a local one-GPU vLLM server, sends all 202 questions directly
to Qwen, and checkpoints generations and judge verdicts under
`outputs/hle-physics-qwen38-direct/`. Set `BENCHMARK_JUDGE=0` to skip semantic
judging.

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

- Add 6.1 as a fallback whenever a model reaches its max output length.
