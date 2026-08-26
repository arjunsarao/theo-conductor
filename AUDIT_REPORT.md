# System audit report

Date: 2026-08-16

## Scope

This audit covered the Python training/reward pipeline, worker execution,
OpenAI-compatible transport, Slurm launcher, tests, and the newest Slurm runs.
The primary evidence came from jobs `19612` and `19609`, with older jobs used
to distinguish current failures from already-resolved startup and GPU-memory
issues.

## Findings

### 1. A judge failure aborted and mislabelled an entire rollout batch

Job `19612` passed the isolated end-to-end preflight and released its CUDA
state correctly, then failed in the first full 256-rollout batch. At least one item
(`rollout-228`) exhausted all six primary/fallback requests with a 600-second
timeout. `asyncio.gather` discarded any successful sibling results, the reward
callback re-raised the selected failure, and the trace error handler attached that
same item ID and six-attempt count to all 255 otherwise judgeable traces.

The resulting trace file therefore reported 255 copies of one judge error even
though the terminal exception identified only `rollout-228`; the old trace no
longer makes the true number of failed sibling requests recoverable. Job
`19609` showed the same batch-wide propagation pattern after a structured
verdict hit its output limit.

Fix:

- Preserve every gathered result and apply success/error metadata to its own
  rollout only.
- Keep an exhausted rollout at the neutral valid-workflow reward (`0.5`) and
  continue training. No heuristic correctness verdict is substituted.
- Keep preflight strict: it still requires a real remote verdict and rejects a
  broken judge configuration before the full run.
- Add a regression test with one success and one failure in the same batch.

### 2. Judge retry policy amplified deterministic and slow failures

The original default could issue three identical Kimi requests followed by
three identical GLM requests. The partial working-tree mitigation raised output
ceilings from 8,192 to 98,304/131,072 tokens. The next run no longer ended on
the same length error, but ended on the much longer timeout failure in `19612`.
The oversized ceilings allowed a tiny JSON verdict to consume a very large
reasoning budget, while each request could remain open for ten minutes.

Fix:

- Use bounded primary/fallback ceilings of 16,384 and 32,768 tokens.
- Default to one explicit attempt per backend and a 300-second request timeout.
  Operators can still raise attempts and budgets through the existing CLI.
- Treat `finish_reason="length"` as deterministic for that backend/budget and
  move directly to fallback instead of repeating the identical request.
- Continue disabling hidden OpenAI SDK retries so the configured policy remains
  authoritative.

### 3. Worker output limits made most reasoning-model calls unusable

In job `19612`, 174 of 256 rollout traces contained at least one empty worker
response. By model, 218/292 Kimi calls (74.7%) and 34/47 GLM calls (72.3%) were
empty; nearly all consumed exactly the old 4,096-token ceiling. The transport
discarded the completion finish reason, so traces could not directly distinguish
normal stops from reasoning-token exhaustion.

Fix:

- Raise the worker-step default from 4,096 to 16,384 tokens.
- Capture `finish_reason` in `ModelResponse`, propagate it to `StepOutput`, and
  persist it automatically in worker-output trace JSON.

This increases the permitted ceiling, not the token use of responses that stop
normally. A new Slurm run is still required to measure the remaining truncation
rate under live model load.

### 4. Dataset transforms were not cacheable

Every preflight and full trainer build emitted a Hugging Face Datasets warning
that `prepare_grpo_dataset` could not be hashed. The map function was a local
closure over `ModelRegistry`, which contains async HTTP clients and is not a
stable serialization target. Dataset caching therefore treated every mapping
as new and recomputed it.

Fix:

- Move row formatting to a module-level function.
- Pass only pre-rendered model lines and examples (plain lists of strings) via
  `fn_kwargs`.

The focused dataset test no longer emits the fingerprint warning.

### 5. Deprecated scheduler configuration and noisy readiness polling

The logs repeatedly warned that `warmup_ratio` will be removed in Transformers
5.2. The Slurm readiness loop also printed one connection-refused line every
five seconds during normal vLLM startup (43 lines in job `19612`).

Fix:

- Resolve the configured ratio to `warmup_steps` with the same ceiling behavior
  used by Transformers (`ceil(max_steps * warmup_ratio)`).
- Silence expected curl connection errors during polling while retaining the
  final timeout/dead-process diagnostic and server-log tail.

### 6. Documentation and local test invocation were inconsistent

The direct CLI defaulted to MegaScience while the current Slurm launcher
defaulted to HLE, and the README did not distinguish them. Running `pytest`
directly also failed collection unless `PYTHONPATH=src` was set manually.

Fix:

- Document the CLI/launcher dataset-default distinction and the new worker and
  judge policies.
- Configure pytest's `pythonpath` in `pyproject.toml`, so the documented source
  layout works with a plain test command.

## Validation

- Full suite: `116 passed`.
- Slurm script syntax: `bash -n scripts/worker_pool.sbatch` passed.
- Python bytecode compilation: `python -m compileall -q src tests` passed.
- Patch hygiene: `git diff --check` passed.
- Focused dataset mapping test passed without the prior fingerprint warning.

Live worker/judge calls and a new GPU Slurm training run were not available in
this audit environment, so the operational fix is validated locally but not yet
cluster-proven.

## Residual risks and follow-up

- The environment uses CPython 3.12.0 with `multiprocess` 0.70.19. That package
  calls a private `RLock._recursion_count()` method absent from this interpreter
  and emits a shutdown-only `AttributeError`. It did not fail tests or Slurm
  training, and patching site-packages would be fragile. Rebuild the environment
  on a current Python 3.12 patch release and re-run the suite.
- The optional fast kernels for the model are not installed, so Transformers
  falls back to its Torch implementation. This is a performance opportunity,
  not the cause of the observed failures.
- Per-item judging sends up to 256 independent requests at concurrency 16. If
  the shared endpoint remains saturated after the bounded-budget fix, lower
  `--judge-concurrency` before raising timeouts or token ceilings.
- Job `19477` failed because another process left only 10.99 GiB free for the
  vLLM server. Later jobs started the server and demonstrated clean isolated
  preflight release, so that older issue is not addressed by further code
  changes here.
