# PRLBench: Fugu Ultra v2 versus Conductor

This comparison runs **four independent attempts per paper per system**, using
Fugu Ultra v2 as the direct solver and GPT-6.1 Sol as the conductor planner with
the current frontier worker pool. Every attempt includes the paper's complete
task bundle. The public
[PRLBench dataset](https://huggingface.co/datasets/AdrianMiao/PRL_Bench) at revision
`c656d7f` contains 100 papers, 357 subtasks and 1,898 rubric criteria across six
subfields. A full run has 400 sample slots per system (800 total).

## Prepare and run

Run from the repository root using the existing Python 3.12 environment.
No Docker installation or image build is needed. The worker interpreter runs
in OpenRouter's hosted sandbox, while the local machine manages API requests
and durable results.

```bash
.venv/bin/python -m theo_conductor.prlbench_comparison prepare \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter

# Requires OPENROUTER_API_KEY in the environment or .env.
.venv/bin/python -m theo_conductor.prlbench_comparison run \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter

.venv/bin/python -m theo_conductor.prlbench_comparison summarize \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter
```

`prepare` downloads a pinned Hugging Face revision. Pass `--source FILE.jsonl`
to use an original local PRLBench file instead; local source provenance records
its content hash without assuming a Hub revision. For an explicit smoke cohort,
prepare a separate directory with `--max-papers 1`. Subsets are marked in the
manifest. Preparation copies the original source, normalized questions, model
registries and planner prompt; it records their hashes, code revision and
implementation hashes. It rejects a nonempty output directory. `run` rejects
changed prepared inputs or implementation; prepare a new comparison after code
changes. Completed historical runs can still be summarized.

Both systems run concurrently and receive the same hosted shell tool, solver
temperature (`0.2`), independent sample count and judge. The conductor keeps its
existing worker catalogue, including Gemini 3.7 Flash; Gemini 3.8 Flash is the
separate judge for both systems. The planner temperature is `0.1`.
Each model uses its registry output limit. `--max-tokens` overrides the solver
and worker limits together. These are configured-system comparisons with
different compute budgets.

For Slurm on a CPU node with API access:

```bash
mkdir -p slurm_logs
sbatch scripts/benchmark_prlbench_comparison.sbatch \
  outputs/prlbench-fugu-vs-conductor-m4-openrouter
```

The job requests 16 CPU cores and 32 GiB RAM, with no GPU. The default is four
concurrent question/sample slots per system. Worker DAGs can have additional
concurrent requests. Extra arguments after the directory are forwarded to the
comparison runner.

## Interpreter design

Workers use the native [OpenRouter hosted shell](https://openrouter.ai/docs/guides/features/server-tools/shell)
through `/api/v1/responses`. The planner and Gemini judge retain their existing
Chat Completions transport. This is a server-side tool: OpenRouter executes the
commands and feeds their results back to the model within the same request.
The benchmark host never executes model-generated code. There is no Spawn
agent or custom local code interpreter. The generic evaluator also exposes
this backend with `model/workflows --openrouter-shell`.

Each worker request gets a fresh random container ID, including each recovery
request and each of the four independent sample slots. Commands within one
request share files and state; workers do not share containers or files.
Workers pass numerical results and derivations through their permitted
predecessor text. The tool explicitly uses `engine: "openrouter"` for both
systems, avoiding a switch to OpenAI's separate native sandbox, and
`network_policy: {"type": "disabled"}`. There is no search, outbound internet,
online package installation, clarification or model delegation in this
benchmark mode. Dataset rubrics stay out of solver and planner requests.
Only the global `https://openrouter.ai/api/v1` endpoint is accepted.

Live probes on October 5, 2026 verified execution for Fugu Ultra v2 and
GPT-6.1 Sol, blocked internet access, and plot download. The observed environment
was Python 3.11.14, NumPy 2.4.6, pandas 3.0.5 and Matplotlib 3.11.1.
**SciPy, SymPy, mpmath, QuTiP and NetworkX were absent.** The prompts declare
this limited package set and allow general shell commands. Package versions
are controlled by OpenRouter and can change; this is not a pinned scientific
image. The previous Docker backend and Dockerfile have been removed.
The provider supports attaching offline files, but this harness does not
upload or install additional dependency bundles.

OpenRouter controls the internal tool loop and command resource limits.
Its documented defaults are 120 seconds per command and 16,384 characters
per stdout/stderr stream, capped at 300 seconds and 65,536 characters.
The model supplies command limits and can issue multiple calls within one
response. **Local `--max-tool-rounds` applies to client tools only; it does
not cap this server-side loop.** This backend does not promise configurable
CPU/memory limits or custom images. Existing model token limits, request
timeouts and declared per-slot recovery attempts still apply.

The benchmark checkpoints the complete Responses output, shell commands,
stdout, stderr, exit/timeout outcomes, provider usage and container/file IDs.
It retains the final assistant response rather than a pre-execution preamble.
Save requested plots as PNG under `~/outputs/`. After generation is saved,
the harness lists every container file (with pagination) to find PNGs,
including those omitted from the provider's ten-file citation limit.
It downloads plots to content-addressed files under each system's `artifacts/`
directory and records SHA-256 hashes. The judge receives actual images and
execution transcripts. These downloads use the host's authenticated Files API;
they do not grant internet access to the worker sandbox.

A failed plot listing/download is a judge error, retaining the successful
solver response. Each inventory and completed download is checkpointed, so
judge-only recovery resumes retrieval without calling a solver again.
Invalid PNGs, hash mismatches and more than 32 MiB of combined image evidence
are judge errors rather than silent omissions. The provider retains remote
container files for 30 days after last use; keep local artifacts with the run.
Only final surviving PNG files are downloaded, not snapshots of every overwritten
intermediate plot. Other remote file citations remain in the raw execution record.

## Cost

[OpenRouter's published price](https://openrouter.ai/blog/announcements/shell-tool/)
is **$0.0001 per sandbox second**, with a **30-second minimum ($0.003)** when
a request starts a new or sleeping container. The clock runs from the first
sandbox command until the response completes, including subsequent model
reasoning in that request. Idle time between requests is not billed. Files API
usage is included. Model token charges and Gemini judging are additional.

For 100 papers × four attempts × two systems, if every attempt uses one new
sandbox, the sandbox minimum is **$2.40**. If the conductor uses seven separate
workers per attempt and all execute shell commands, 400 Fugu requests plus
2,800 conductor requests imply **$9.60** at the minimum. These are conditional
minimums, not estimates of final charges: active time above 30 seconds and
recovery attempts add cost; requests without shell execution incur no sandbox
charge. A sandbox active for one minute costs $0.006, or $4.80 for 800 such
requests, before model and judge tokens.

The compatibility probes cost **$0.0812729 total**, including $0.0093894 of
sandbox charges. Their usage and raw responses are saved in
`outputs/prlbench-openrouter-compatibility/`. These short probes cannot predict
the token cost of a full research benchmark.

Two additional live checks through the implemented adapter verified numerical
computation, PNG creation and complete download for both models. They cost
**$0.1291763 total**, including $0.0147483 in sandbox charges, with evidence in
`outputs/prlbench-openrouter-native-smoke/`. All four development probes together
cost **$0.2104492**. The full comparison has been prepared in
`outputs/prlbench-fugu-vs-conductor-m4-openrouter/` but has not been launched.

Usage summaries retain provider `cost` as the inclusive request charge and
report `server_tool_cost_usd` and `upstream_inference_cost_usd` separately where
provided. Sandbox costs are not added to the inclusive charge again.
List-price model token estimates remain separate. Failed/interrupted requests
with unavailable usage are unknown, not zero.

## Judging and metrics

Both systems use OpenRouter's
[Gemini 3.8 Flash](https://openrouter.ai/google/gemini-3.8-flash), registry ID
`gemini-3.8-flash` and provider slug `google/gemini-3.8-flash`. Judging uses
temperature zero, a 32,768-token limit, and strict JSON output with one boolean
and explanation per rubric criterion. Missing, duplicate or unexpected IDs,
nonboolean verdicts and empty reasons are rejected. Judging considers the full
final response, extracted answer, current successful-step execution evidence
and saved plots. Rubrics are sent only to the judge.

The public file contains rubric statements but no numeric point weights or
separate weighted answer entries. The
[paper](https://arxiv.org/html/2604.15411v1) describes predefined weighted scores;
this harness therefore uses an explicit adaptation: each public criterion
counts equally within its paper, and paper scores are averaged equally.
Every criterion's substantive requirements must be met to receive credit.
This is not the paper's official weighted score or an exact reproduction of
its five-repeat, GPT-5 judging protocol. Search is disabled, consistent with
the paper's tool setting.

An attempt passes when **at least 70% of its criteria are met**, as requested.
`prepare --pass-threshold FRACTION` declares a different policy in a new
comparison. `comparison-summary.json` reports:

- `mean_at_4`: mean binary success over four independent attempts per paper.
- `pass_at_4`: fraction of papers with at least one successful attempt, an
  oracle metric rather than an answer-selection procedure.
- `mean_criterion_score_at_4_percent`: average raw criterion satisfaction score
  over the four attempts and all papers, on a 0–100 scale.
- Subfield mean@4/pass@4, coverage, failure counts and complete evaluation
  usage/cost/latency summaries, including planner, worker, tool, judge and retries.

All expected papers and slots remain in the denominator. Exhausted solver or
workflow failures count as false (and zero raw score). Missing plans, missing
slots and judge errors remain unknown; overall metrics stay null until the
cohort is complete. The report rejects incompatible inputs, registries,
grading policies, sample counts or paired interpreter settings.

## Resume and judge-only recovery

Repeat `run` with the same output directory to resume. Each system uses the
existing `theo-evaluate` managed checkpoints. `--attempts` defaults to three
recovery attempts within a sample slot; retries never increase k. Successful
worker ancestors and siblings are retained. Changing generation settings
invalidates affected generations and grading, while preserving attempt history.
API interruptions cannot guarantee exactly-once billing.

To recover judging and pending plot downloads without invoking either system:

```bash
.venv/bin/python -m theo_conductor.evaluate judge \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter/fugu \
  --config outputs/prlbench-fugu-vs-conductor-m4-openrouter/fugu.yaml \
  --judge-model gemini-3.8-flash --judge-max-tokens 32768

.venv/bin/python -m theo_conductor.evaluate judge \
  --output-dir outputs/prlbench-fugu-vs-conductor-m4-openrouter/conductor \
  --config outputs/prlbench-fugu-vs-conductor-m4-openrouter/conductor.yaml \
  --judge-model gemini-3.8-flash --judge-max-tokens 32768
```

Use `--force` only to replace existing judge verdicts. Keep the run's plot files
with its checkpoints for rejudging. Finish by repeating `summarize`.
