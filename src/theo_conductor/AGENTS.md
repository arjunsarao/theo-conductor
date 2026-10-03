# theo-evaluate agent guide

## Scope

Inherit the [repository guide](../../AGENTS.md). This file is located at the
Python package boundary because `theo-evaluate` lives in `theo_conductor`.
The evaluation-specific rules below apply to `evaluate.py`,
`evaluation_stats.py`, `jsonio.py`, and changes to shared modules that affect
evaluation. They do not replace training's separate reward policy.

`theo-evaluate = theo_conductor.evaluate:main` is registered in `pyproject.toml`.
Prefer this entry point for new resumable runs; existing provider batch launchers
have their own lifecycle and remain supported. See the
[resumable evaluation documentation](../../README.md#resumable-evaluation).

## Entry points and implementation

```bash
.venv/bin/python -m theo_conductor.evaluate --help
.venv/bin/python -m theo_conductor.evaluate catalog --root outputs
.venv/bin/python -m pytest tests/test_evaluate.py
```

Run these from the repository root. `plans` generates plans only; `workflows`
executes stored plans or generates and executes new ones; `model` evaluates one
registry model; `judge` grades saved completions without generation; `catalog`
lists runs; `aggregate` combines compatible runs. `--force` replaces judge
verdicts, not solver generations. Model flags take registry IDs.

- `evaluate.py`: CLI, `RunStore`, request fingerprints, execution, import,
  judging, and aggregation.
- `evaluation_stats.py`: usage summaries and repeated-sampling metrics.
- `jsonio.py`: validated JSONL and fsync/atomic snapshot writes.
- `runner.py`: dependency-aware reuse, worker failure detection, and request/tool
  checkpoint hooks. `benchmark.py`: shared judge payload and strict verdict parser.
- `../../scripts/evaluate.sbatch` and `../../scripts/lib/evaluation_env.sh`:
  cluster entry point and shared environment setup.

## Durability and resume invariants

- Validate nonempty questions and unique IDs. Each independent observation is
  identified by `(example_id, sample_index)`; recovery attempts stay in that slot.
- Keep the nonblocking exclusive run lock. Reject incompatible input/model/mode
  identities, decreasing sample counts, and unmanaged legacy results in a managed
  output directory. Import legacy completions into a new directory.
- `checkpoints/*.json` are authoritative after interruption. Preserve atomic
  replacement and file/directory fsync in `jsonio.py`. Rebuild exported
  `results.jsonl`, `plans.jsonl`, `summary.json`, and `failures.jsonl` from state.
- Reuse plans, worker outputs, solver responses, and judge verdicts only when
  the relevant request fingerprints match. Include all behavior-affecting inputs
  when adding options: model/provider, prompt, image, context, limits, and sampling
  settings as applicable. A changed dependency invalidates its descendants.
- A failed/capped step may retry while successful ancestors and independent
  siblings remain reusable. Persist each request/tool round and successful step;
  record interrupted requests without inventing their unknown usage.
- Reject capped (`length`/`max_tokens`), empty, errored, and filtered completions
  as successful generation. Dependents must wait for usable upstream outputs.
- Preserve attempt history and clear stale grading when candidate output changes.
  Judge-only recovery must not invoke planner, solver, or worker generation.
- Do not promise exactly-once requests/billing after interruption. External tool
  side effects need their own idempotency support when a step is replayed.

## Grading and statistics

- Use the full candidate response plus extracted final answer for semantic
  judging. Missing `FINAL:` is a diagnostic flag, not an automatic wrong answer.
- Require a valid verdict ID, boolean correctness, and nonempty reason. Preserve
  strict rejection of missing, duplicate, or unexpected verdict IDs. Parsing/API
  errors remain `judge_error` with unknown correctness rather than a guessed false.
- Keep exhausted solver/workflow failures in the sample denominator as false
  under the current evaluation policy. Missing plans and judge errors remain
  unknown; repeated metrics stay null until all expected outcomes are available.
  Plan-only runs have no correctness metrics or demonstrated solver rewards.
- `--samples` defines independent slots; `--attempts` defines recovery attempts.
  Never count retries as additional k samples. Reject duplicate slots and
  incompatible input/model/generation cohorts during aggregation.
- Preserve `mean_at_k` as average sample correctness and `pass_at_k` as
  `1 - C(n-c,k)/C(n,k)` averaged over questions. At k=n, pass@k is the fraction
  solved at least once. It is an oracle metric, not deployed answer selection.
- Keep unknown usage distinct from zero and report known/unknown counts. Separate
  provider charges from list-price estimates and source/imported request costs
  from requests issued in the current run. Include planner, worker, solver, tool,
  judge, and recovery attempts when reporting observed end-to-end cost.

## Historical findings to preserve

The [October 2 HLE-Diamond audit](../../reports/hle_diamond_run_audit.md) and its
[evidence JSON](../../reports/hle_diamond_audit_evidence.json) and
[flag CSV](../../reports/hle_diamond_audit_flags.csv) are the main references.

- The historical conductor cohort had 23 capped worker outputs out of 569,
  affecting 23 of 244 workflows, plus two empty final answers without a workflow
  error. `Runner.step_failure` and strict execution now detect incomplete outputs;
  that does not retroactively repair these runs. Report worker failures separately
  from top-level workflow completion and verify safeguards in fresh runs.
- Six conductor image-routing failures occurred despite images being forwarded.
  Validate selected endpoint image support; retaining an image in a record does
  not prove that the selected worker can consume it. Do not drop image questions
  to repair scores. The shared judge payload does not forward images, so review
  image-dependent disputes with suitable evidence or an image-capable judge.
- The original Claude route failed all 244 attempts with HTTP 402. Exclude it
  from model-capability comparisons and label the successful Anthropic replacement
  as a different route/settings cohort. Infrastructure failure rates can be
  reported as end-to-end reliability, separately from capability conditional on
  successful requests.
- Today's prompt/configuration differs from historical runs. Retain manifests,
  request histories, prompt/config hashes, dataset identity, model/provider route,
  budgets, sampling parameters, and code revision when available. Explain missing
  historical provenance rather than inferring it from current YAMLs.
- Recover caps under a declared policy covering all capped cases, or use a fresh
  full cohort. Preserve the original results, regenerate affected descendants,
  and rejudge changed final answers. Selectively retrying only incorrect capped
  cases biases comparisons; increased budget need not improve each answer.
- The audited cohort covers 61 HLE-Diamond Physics questions repeated four times,
  not 244 independent questions or all HLE subjects. Compare paired question-level
  outcomes; bootstrap whole questions with their repeats together. Equal pass@4
  counts do not establish equal solved sets, cost, or accuracy. Configured-system
  comparisons are not automatically compute-matched orchestration experiments.
- The [earlier error analysis](../../reports/gpt6_conductor_error_analysis.md)
  found summary/record disagreement and disputed reference answers. Recompute
  from source records, disclose unknown verdicts, and flag questionable references
  for independent review without silently rewriting gold answers or awarding credit.

## Verification expectations

For evaluation changes, use `tests/test_evaluate.py` fake clients and temporary
directories. Cover the relevant behavior: resume after failure/interruption,
successful sibling retention, cap recovery, dependency invalidation, judge-only
recovery, sample extension, identity/lock rejection, legacy import, or unknown
usage and cohort aggregation. Run related runner, prompt, transport, or benchmark
tests when those contracts change. Keep local regression checks free of paid
requests; live evaluation requires the user's task to authorize those runs.
