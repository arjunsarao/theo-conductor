# GPT-6-as-conductor worker-output error analysis

## Scope and counting basis

Analyzed `outputs/hle-physics-gpt6-frontier/results.jsonl` (202 workflows) against the plans and every stored worker output.

The record-level judge fields contain:

- 99 `judge_correct=true`
- 101 `judge_correct=false`
- 2 `judge_correct=null` because the judge response could not be parsed

This differs from `summary.json`, which reports 101 correct and 101 incorrect. The analysis below uses the 101 explicit negative judge verdicts. If execution/evaluation failures are included, there are 103 records whose top-level `correct` field is false.

## Quantitative findings

The following counts are diagnostic dimensions and overlap; they should not be added together.

| Diagnostic | Count | Interpretation |
|---|---:|---|
| Judged errors | 101 | Explicit `judge_correct=false` records |
| Errors from one-step workflows | 56 | No independent worker or downstream verifier could correct the sole worker |
| Errors from two-step workflows | 18 | A solve/check or solve/final pattern failed |
| Errors from three-step workflows | 27 | Candidate/adjudication/synthesis still failed |
| Final answer explicitly rejected the premise as non-unique, underdetermined, inconsistent, or without a listed answer | 20 | A large, distinctive “over-skepticism / benchmark-intent” failure mode |
| Of those 20, conductor instructions explicitly foregrounded ambiguity, missing assumptions, or non-uniqueness | 17 (lexical lower bound) | The plan itself primed the refusal; manual review suggests the same dynamic in most of the remaining three |
| Multi-step errors where an earlier worker surfaced the gold/intended answer or the decisive correction, but the final worker rejected or failed to use it | 9 | Direct aggregation/adjudication failures; an objection channel alone would not have been sufficient because the objection already existed in text |
| Empty worker response | 1 | Provider/model execution failure (`benchmark_position=144`) |
| Strongly reference-disputed or materially under-specified items found in a conservative internal screen | 26 | These should not be treated as clean system errors without re-adjudicating the benchmark |

Workflow error rates among the 200 parseably judged records were:

| Steps | Judged | Wrong | Error rate |
|---:|---:|---:|---:|
| 1 | 91 | 56 | 61.5% |
| 2 | 50 | 18 | 36.0% |
| 3 | 59 | 27 | 45.8% |

This is observational, not causal: the conductor routed different difficulty profiles to each topology. Still, the 56 one-step failures show that the planner frequently left no error-correction opportunity. In the medium-difficulty stratum, one-step workflows were wrong on 44/76 (57.9%) while two-step workflows were wrong on 14/36 (38.9%).

## Direct answer to the objection-channel hypothesis

There are two different failure modes:

1. **No opportunity to object.** In 56 judged errors the plan had one worker, so there was no independent critic at all. Parallel workers also cannot see or respond to each other; only a later step with both outputs in its `access_list` can reconcile them.
2. **Objection existed but was not operationalized.** In at least 9 multi-step failures, a worker already stated the benchmark's intended answer or decisive correction, yet the final worker selected another conclusion. These are not principally caused by the absence of a free-text objection mechanism. They are caused by weak aggregation: objections have no structured status, confidence, claim-level evidence, or requirement that the final worker explicitly resolve them.

High-confidence examples of the second mode:

| Benchmark position | What the worker caught | What final did |
|---:|---|---|
| 14 | The interpretation worker recommended choice A | Final selected E after emphasizing missing experimental details |
| 50 | The audit worker identified G as the likely intended answer | Final refused all options as non-universal |
| 76 | The independent worker computed the requested bosonic-sector count 6 | Final counted the full commuting body and returned 8 |
| 140 | The comparison worker identified case 3 as the strongest flux guide | Final explicitly rejected case 3 |
| 175 | The numerical worker produced approximately 0.295 and 1.36 | Final refused to state the requested three-digit minima |
| 178 | An earlier derivation contained the gold normalization as its trace-2 convention | Final chose the factor-of-two alternative |
| 179 | The first worker gave the gold closed form under the trace-2 convention | Final chose the orthonormal convention and doubled it |
| 186 | The dynamics worker selected F and matched its bracket | Final rejected every choice because of a flux-factor objection |
| 194 | The worker derived the gold nontrivial branch value | Final included it but declared the problem non-unique because of the zero branch |

Position 30 is an additional near-example: the first worker said a decrease in `N_eff` was physically likely, but the final worker elevated a caveat into “not uniquely determined.” It is excluded from the count of 9 because the earlier worker did not state the conclusion unconditionally.

## Root causes

### 1. Static, one-shot planning and under-orchestration

The conductor emits the complete DAG before any work is performed and is not called again. A one-step workflow therefore has no independent check, and a multi-step workflow cannot add a targeted worker when an unexpected disagreement appears.

This is the largest directly measurable architectural exposure: 56/101 judged errors were one-step plans. Eleven of fourteen conductor-labeled “easy” questions failed, suggesting that difficulty/routing calibration was especially poor at the easy end.

### 2. Conductor-induced framing and assumption anchoring

The worker system prompt says to follow the assigned instruction exactly. Many conductor instructions did more than identify a risk: they foregrounded a particular skeptical interpretation, such as “determine whether the question is uniquely specified,” “do not silently assume,” or “if underdetermined, say so.”

That framing produced 20 final refusals/non-unique answers among the 101 judged errors. In 17, the ambiguity frame is explicit in the stored conductor instruction by a conservative lexical test. The issue is not that checking assumptions is bad; it is that the plan lacks a countervailing requirement to infer the conventional/intended interpretation and answer it when the task is clearly benchmark-like or multiple-choice.

### 3. Unstructured disagreement handling and weak final synthesis

Worker outputs are opaque prose blobs. There is no schema for:

- proposed answer;
- assumptions;
- confidence;
- detected contradiction with another worker;
- “blocking objection” versus optional caveat;
- evidence or a check that would resolve the disagreement.

The final worker is merely instructed to synthesize. It can ignore a correct minority worker, prefer an over-cautious interpretation, or select the wrong normalization. The 9 cases above are a lower bound on this failure mode.

### 4. No post-final verification or answer-contract check

The runner only appends `FINAL: <answer>` to the last instruction. It does not verify that the final answer:

- selects one of the supplied multiple-choice options;
- returns the requested number rather than a caveat;
- is consistent with its own derivation;
- uses the convention stated in the question;
- resolves every explicit disagreement in predecessor outputs.

Examples include position 54, where the worker's derivation matched option E but it selected B; position 86, where the worker listed symmetry factors 48, 16, and 128 but summed reciprocal weights instead of the requested factors; and positions 178/179, where the correct normalization appeared in the analysis but the final line chose the other convention.

### 5. Tool and retrieval constraints

Every worker was explicitly told that no external tools or code execution were available. This predictably hurt exact lookup and numerical tasks. Clear examples include the exact IAU constellation-boundary endpoints (position 187), a diffraction peak that requires pattern/structure data (167), and numerical bootstrap optimization (175). This is a real system limitation, although it does not explain most algebraic/conceptual errors.

### 6. Worker technical errors or shared misconceptions

In many remaining traces all workers make the same sign, factor, model-selection, or convention mistake. Adding workers does not help when their errors are correlated, especially when the conductor gives all of them the same framing. Examples include the factor-of-two cross-section miss at position 20 and the mechanics expression at position 130.

### 7. Benchmark and evaluation defects

At least 26 negative verdicts deserve re-adjudication before being used to diagnose the conductor. This is a conservative internal screen, not a claim that all 26 gold answers are definitively wrong. The strongest examples are:

- position 109: the standard one-loop correction to `nu` is `O(epsilon)`, while the gold says `O(epsilon^2)` (which resembles the order for `eta`);
- position 110: the response gives the standard leading Ising `phi^4` result `alpha=epsilon/6`, while the gold says `epsilon/2`;
- position 139: the judge itself acknowledges the worker's circuit derivation gives `1/4`, but the gold is 0;
- position 145: the worker gives the standard `J^2=j(j+1)hbar^2=15/4 hbar^2` for `j=3/2`, while the gold expects 3.5;
- position 157: `Z1Z2`, `Z2Z3`, and `Z3Z4` are commuting independent stabilizers whose joint +1 space is exactly the span of `|0000>` and `|1111>`; the reference rationale incorrectly says the logical operator is in the stabilizer group;
- position 196: the differential equation itself forces `y'=sin(y-t)-4` to lie in `[-5,-3]`, so the gold `-32/13`, approximately `-2.46`, is impossible;
- position 200: the judge rationale is internally inconsistent about the factor used for `X2`.

Other strong under-specification/reference flags in the conservative set are positions 5, 8, 23, 36, 44, 50, 66, 71, 74, 77, 98, 101, 106, 107, 111, 125, and 181–183. These cases often reward an unstated textbook convention while penalizing a technically defensible objection.

## Recommended system changes

1. Add a structured worker result with `answer`, `assumptions`, `confidence`, `objections`, and `verification_needed` fields.
2. Add a blocking-objection state that triggers a conductor replan or a targeted adjudicator. Do not let it silently terminate the DAG.
3. Require the final worker to enumerate disagreements and state why each losing claim is rejected before emitting the final answer.
4. Add a cheap post-final verifier for answer type, multiple-choice membership, requested precision, internal consistency, and convention matching.
5. Change ambiguity policy: give the conventional/intended answer first when recoverable, then state caveats. Reserve outright refusal for genuinely non-identifiable answers.
6. Calibrate topology using observed risk. The current “easy → one worker” policy is unreliable on this dataset; use at least a lightweight verifier for benchmark questions with numerical, multiple-choice, or convention-sensitive answers.
7. Give selected workers code execution/retrieval for numerical optimization and exact factual lookup, with citations or reproducible calculations.
8. Re-adjudicate disputed benchmark items with domain experts or independent primary-source checks before using aggregate accuracy to tune the conductor.
