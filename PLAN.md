# Theo Conductor Plan

## Readiness Check

`train.py` is no longer just a sketch. It already:
- loads a dataset from Hugging Face,
- builds conductor prompts,
- loads a model processing class,
- constructs a `GRPOTrainer`,
- and can start training once the remaining pieces are in place.

What is still missing is the set of pieces that make the run meaningful and reliable.

## Must Finish Before First Real Training Run

1. Verify the training entrypoint end to end.
- Run `src/theo_conductor/train.py` in `--dry-run` mode.
- Confirm the prompt format matches what the conductor model should emit.
- Confirm the dataset rows contain the fields the trainer and reward function expect.

2. Finalize the conductor output format.
- Decide whether the model should emit raw JSON, fenced JSON, or a relaxed JSON-ish format.
- Make the parser in `src/theo_conductor/grpo.py` match that decision.
- Ensure the parser always produces a valid `Task` or a clear failure reason.

3. Confirm the reward function is usable for GRPO.
- Keep the basic reward ladder:
  - `0.0` malformed output
  - `0.2` parseable but invalid workflow
  - `0.5` valid workflow but wrong answer
  - `1.0` correct answer
- Decide whether to include extra penalties for cost and latency in the first version.
- Verify the reward signature matches the TRL version in use.

4. Make worker-model access reliable.
- Confirm every worker in `configs/frontier_models.yaml` has valid credentials and a working endpoint.
- Confirm local worker configs still load through `ModelRegistry`.
- Decide which worker pool is the actual training target for the first run.

5. Validate the task execution loop.
- Confirm the conductor output can be parsed into a `Task`.
- Confirm `Runner` can execute that task with the configured worker registry.
- Confirm the final answer can be extracted consistently.

6. Fix repository mismatches that block confidence.
- Reconcile tests and code paths that still refer to missing local dataset wrappers.
- Make sure `main.py` and any other entrypoints do not contain dead or un-awaited async calls.
- Remove or rewrite any stale comments or TODOs that no longer describe the code.

## Next After the First Run Works

1. Add a proper evaluation set.
- Use a held-out subset of HLE, GPQA, or MegaScience.
- Track exact-match, normalized answer match, and task validity separately.

2. Add stronger grading.
- Numeric tolerance checks for physics.
- Symbolic equivalence where possible.
- Optional LLM judge for derivation quality.

3. Add baseline comparisons.
- Single best worker.
- Best-of-N worker sampling.
- Parallel adjudication.
- Template workflow baseline.

4. Add training observability.
- Save trajectories.
- Log costs and latency.
- Save parser failures and invalid workflows.
- Record per-worker success rates.

5. Improve the conductor prompt.
- Add worked examples of valid workflows.
- Add explicit rules for tool use and access lists.
- Add format constraints that reduce malformed JSON.

## Suggested Execution Order

1. Dry-run `train.py`.
2. Validate worker registry access.
3. Run one tiny end-to-end rollout.
4. Verify reward computation on a handful of examples.
5. Launch a short GRPO training run.
6. Inspect failures and tighten the prompt/parser/reward loop.
