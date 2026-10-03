# theo-conductor agent guide

## Scope and orientation

This repository is the Python `theo-conductor` package. It generates conductor
plans, executes worker DAGs, benchmarks model responses, and trains planners
with GRPO. `theo-evaluate` is a console entry point in this package, not a
separate distribution. Read [README.md](README.md) for operational commands and
[src/theo_conductor/AGENTS.md](src/theo_conductor/AGENTS.md) for evaluation rules.

Treat the current source, configuration, and tests as authoritative for current
behavior. Reports describe particular historical runs; their recommended
changes are not proof that those changes have been implemented.

## Repository map

- `src/theo_conductor/prompt.py`, `plan.py`: planner prompts, constrained JSON
  output, validation, and saved-plan inspection.
- `schema.py`, `validate.py`, `scheduler.py`, `runner.py`: workflow types,
  dependency validation, scheduling, worker execution, and checkpoints.
- `models/`: registry and model clients; `tools/` and `artifact.py`: worker
  tools, clarification, and artifacts. Paths in this list are under
  `src/theo_conductor/` unless otherwise indicated.
- `evaluate.py`, `evaluation_stats.py`, `jsonio.py`: resumable evaluation and
  shared durable storage/statistics. `benchmark.py`, `workflow_benchmark.py`,
  `diamond_benchmark.py`, and `anthropic_benchmark.py` support existing workflows.
- `train.py`, `grpo.py`, `data.py`: training, rewards, and dataset preparation.
  `traces.py`, `trace_analysis.py`, and root `trace_viewer.py`: trace inspection.
- `conductor-prompt-refined.md` is the current default planner template;
  `worker-prompt.txt` is the worker template. `configs/` holds registry YAMLs.
- `scripts/` contains analysis and Slurm launchers; `tests/` contains local
  regression tests; `reports/` contains historical analysis and evidence.
- `outputs/`, `data/`, `slurm_logs/`, and `wandb/` are local run artifacts.

## Development and verification

Use Python 3.12 (`>=3.12,<3.13`) and the existing `.venv` when available.
Dependencies and CUDA wheel indexes are pinned in `pyproject.toml`/`uv.lock`;
avoid incidental dependency upgrades or installing GPU stacks for a docs edit.

Run commands from the repository root:

```bash
.venv/bin/python -m pytest tests/test_prompt.py tests/test_validate.py tests/test_simple.py
.venv/bin/python -m pytest tests/test_evaluate.py
.venv/bin/python -m pytest
git diff --check
```

Select tests for the changed behavior; use the full suite for changes spanning
shared execution, transport, schemas, or training. Pytest already has `src` on
its import path. Tests should use fake clients and temporary output directories.
For Slurm/bash edits, run `bash -n` on the changed scripts. For documentation-only
changes, check paths, commands, and patch whitespace; model/GPU runs are unnecessary.
State which checks ran and distinguish local verification from live validation.

Preserve unrelated working-tree changes. Edit active files rather than `.orig`
backups. Keep secrets out of source, reports, and tool output; do not print `.env`.
Use existing artifacts and fake clients for development. Submit paid model calls,
GPU training, or Slurm jobs only when they are within the user's requested scope.

## Workflow and model contracts

- Keep planner JSON schema, prompt template, runtime validation, and tests in
  agreement. Compact plans have `task_type`, `difficulty`, and 1–7 workflow steps;
  each step has `step_id`, `model_id`, `instruction`, and `access_list`.
- Model arguments are registry IDs (`model_idx` in YAML), not provider slugs.
  The prompt's worker catalogue must exist in the chosen registry. Retain exactly
  one `<USER QUESTION>` marker in the planner template.
- Preserve context access boundaries and dependency order. A worker sees the
  inputs granted by its access list; question/image forwarding must remain
  consistent across planner, direct solver, and worker routes.
- Preserve raw worker text, finish reasons, usage, tool calls, and error metadata.
  Respect configured per-model output limits and explicit overrides. High token
  use alone does not establish truncation; inspect finish reasons and text.
- Keep async failures local to the affected item/step and retain successful
  siblings. Do not let one failed judge request relabel an entire batch.
- Keep benchmark gold answers and reference rationales out of solver/planner
  requests. They belong to judging and recordkeeping.

## Lessons from previous reports

Read these before changing the corresponding subsystem:

- [Conductor error analysis](reports/gpt6_conductor_error_analysis.md): among
  101 explicit negative verdicts, 56 came from one-step plans and nine multi-step
  cases lost a useful earlier candidate or correction at synthesis. These are
  overlapping, observational diagnostics, not causal topology comparisons.
  When changing orchestration, test whether final synthesis resolves predecessor
  disagreements and preserves decisive corrections.
- [Refusal analysis](reports/gpt6_refusal_failure_analysis.md): planner framing
  sometimes primed workers to reject recoverable benchmark questions. Prefer
  solving a defensible conventional interpretation, stating assumptions, and
  returning the requested value/option. Require concrete evidence for fatal
  ambiguity. Structured objections and answer-contract repair are proposals;
  do not describe them as existing capabilities without checking the code.
- [Rollout similarity](reports/rollout_similarity_21510.md): the 64-rollout run
  contained plans only, with neutral placeholder rewards and no worker execution.
  A reduction to 32 is an ablation candidate, and 16 is more aggressive. Similarity
  alone cannot justify changing GRPO defaults: evaluate held-out reward, reward
  variance, zero-advantage groups, gradient stability, and effective rollout cost.
- [System audit](AUDIT_REPORT.md): preserve per-rollout judge outcomes, bounded
  retries/fallbacks, and finish-reason capture. Training's neutral reward for an
  exhausted judge is not an evaluation correctness verdict. Keep preflight strict.
  Dataset mapping functions should accept serializable data rather than close
  over registries containing async clients. Reported fixes were locally verified;
  historical test counts and budgets are not current defaults.
- [HLE-Diamond audit](reports/hle_diamond_run_audit.md): workflow completion counts
  hid intermediate caps and empty finals, and image-incompatible routing caused
  failures. Inspect worker-level evidence and endpoint capabilities. Current
  safeguards do not repair historical artifacts. Follow the evaluation guide for
  recovery, grading, provenance, and comparative claims.

When producing a new analysis, identify input files/cohorts, denominator, failure
policy, and limitations. Preserve original evidence and distinguish observed
findings, hypotheses, proposed fixes, and verified implementation.
