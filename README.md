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
and raw conductor completion.

From the repository root, run:

```bash
uv run streamlit run trace_viewer.py
```

The app loads the default `outputs/grpo-11352` rank-0 trace automatically. Use
the sidebar to open a repository path, load a SLURM job by ID, or upload any
JSONL trace. A repository trace can also be selected with
`?trace=path/to/trace.jsonl`. The original `trace_viewer.html` remains available
as a dependency-free viewer when served from the repository root.

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
worker workflows during rewards, use `scripts/small_local_model_grpo.sbatch`
or pass `--execute-workflows` to `python -m theo_conductor.train`.

Executed-workflow training uses Kimi K2.6 as the sole semantic correctness
judge. Every valid rollout passed to one GRPO reward callback is packed into a
single judge request; malformed or invalid workflows retain their structural
reward without being answer-judged. The complete request is retried on API or
schema-validation failures, and training stops if all attempts fail—there is no
local exact/numeric heuristic fallback. Configure the endpoint with
`KIMI_BASE_URL`, `KIMI_API_KEY`, and `KIMI_MODEL`; tune failure handling with
`--judge-attempts`, `--judge-retry-delay-seconds`, `--judge-max-tokens`,
`--judge-connect-timeout-seconds`, and `--judge-timeout-seconds`. Judge clients
disable the OpenAI SDK's internal retries so `--judge-attempts` is the exact
number of batch attempts recorded in training traces.

## Small-model MegaScience benchmark

Benchmark every model in `configs/local_small_models.yaml` with one independent
call on the same deterministic 200-row MegaScience validation subset used by
training:

```bash
sbatch --export=ALL,BENCHMARK_MEGASCIENCE=1 scripts/slurm_local_small_models.sbatch
```

The job starts all three vLLM endpoints, verifies them, and writes resumable
per-question records to `outputs/megascience-small-models/results.jsonl` and
aggregate metrics to `outputs/megascience-small-models/summary.json`. Metrics
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
`--seed`, `--total-samples`, and `--validation-samples`; their defaults (42,
2000, and 200) intentionally match conductor training.

# TODO

- Train the conductor on a single set of models using GRPO on the MegaScience Dataset
    1. Use DeepSeek-R1-Distill-Qwen-32B as the physics solver.
    2. Use `Gemma-4-31b-it`as a math solver. (use the 31B dense model)
    3. Use Qwen2.5-Coder as a coder.
- Create custom system prompts for conductor and ensemble models.
- Implement runner/dispatch for various llms (make it configurable)
- Add HLE benchmarking from huggingface.
- Question field is added by me, not echoed by the LLM
- Qwen's context length is >200k tokens.

Final filtered data is 1830 questions.

Total calls = 1830 questions * 5 steps per workflow * 4 rollouts
