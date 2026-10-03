# HLE-Diamond Physics run audit

Audit date: October 2, 2026. This is an audit of the local saved artifacts, not a new evaluation or rejudging. No benchmark results were changed and no paid requests were made.

Publication assessment after follow-up: no observed catastrophic grading or aggregation failure. Excluding the superseded Claude HTTP 402 cohort, all **1,697/1,697 judgeable responses have a parseable saved verdict with a boolean decision and nonempty reason**, consistent with `judge_correct`. The other 11 of 1,708 attempts have failed or empty generations and receive policy-assigned false outcomes. Thus missing judge finish reasons are a provenance limitation, with no observed missing-verdict problem. Successful extraction cannot rule out a capped judge response that nevertheless contained a complete verdict. The superseded Claude failures do not affect the replacement cohort and are not a blocker for publishing that cohort.

Before publication, recover capped steps under a declared policy, regenerate every downstream step that depends on a changed output, and rejudge the resulting final answers. Preserve the original results. Additional generation budget does not guarantee monotonic accuracy: a regenerated intermediate answer may change downstream reasoning in either direction. Fugu Max's nine capped direct attempts can improve its mean@4 by at most 9/244 = 3.69 percentage points, to 27.87%, if every one becomes correct; this cannot close its gap to the conductor's recorded 55.74% mean@4. Its pass@4 improvement is also bounded by at most nine questions, from 24/61 to at most 33/61, below the conductor's 48/61. These bounds cover the observed caps, not an unrestricted full rerun at changed settings.

**There are methodological concerns.** The largest are hidden conductor worker truncation, image-incompatible worker routing, and an entirely failed original Claude route. The retained direct Fugu runs also contain capped answers. Dataset coverage and aggregate arithmetic check out; grading infrastructure errors have been resolved in the saved results.

## Scope and results

Inspected all 32 HLE-Diamond `results.jsonl` files: 1,952 question attempts, including the four failed original Claude runs. Also inspected 244 conductor plans and their request records, 569 retained conductor worker outputs, all seven saved aggregate summaries, generation/judging code, launch scripts, and relevant Slurm logs. Classification outputs are auxiliary and are not solver runs.

Every run uses the same 61 Physics questions: 40 text-only, 21 with embedded images; 47 reasoning and 14 knowledge partition questions. Astra's first repeat is the separate `hle-diamond-astra-gemini-batch-22621` directory. It must not be counted again as a separate four-repeat cohort.

| Cohort (four repeats each) | Correct / 244; mean@4 | pass@4 | Direct capped outputs | Other concerns |
| --- | --- | --- | --- | --- |
| Astra | 169; 69.26% | 48/61; 78.69% | 0 | 3 generation failures |
| GPT-6.1 Sol direct | 146; 59.84% | 44/61; 72.13% | 0 | No saved generation failures |
| Claude Fable 5.1, Anthropic batch replacement | 142; 58.20% | 45/61; 73.77% | 0 | Different provider route and generation settings |
| GPT-6.1 Sol conductor + frontier | 136; 55.74% | 48/61; 78.69% | See worker audit below | 23 worker caps; 6 image-routing failures; 2 additional empty final answers |
| Fugu Ultra v2 | 121; 49.59% | 39/61; 63.93% | 1/244 (0.41%) | 3 additional nonempty responses lack an extracted FINAL line |
| JevRouter | 110; 45.08% | 40/61; 65.57% | 0 | No saved generation failures |
| Fugu Max | 59; 24.18% | 24/61; 39.34% | 9/244 (3.69%) | All 9 capped answers graded incorrect |
| Original Claude OpenRouter route | 0; 0% | No aggregate saved | Unobservable | All 244 attempts failed with HTTP 402 insufficient credits; exclude from model ability comparisons |

“No caps” means no cap finish reason in retained successful completions. It does not establish that every earlier request or internal subcall completed normally. Multi-agent/router usage can exceed a caller's token limit because usage may include internal work; token totals alone are not a truncation test.

## Concerns requiring qualification or correction

### 1. Conductor caps and empty/error workers were hidden by workflow completion counts

| Conductor repeat | Retained workers | Capped workers | Explicit workflow failures | Empty final answers without a workflow error |
| --- | --- | --- | --- | --- |
| 1 | 141 | 7 | 2 | 0 |
| 2 | 138 | 3 | 3 | 0 |
| 3 | 143 | 5 | 1 | 1 |
| 4 | 147 | 8 | 0 | 1 |

**23/569 retained worker outputs (4.04%) ended with `length`, affecting 23/244 workflow attempts (9.43%).** Twenty are DeepSeek V4 Pro at exactly 40,960 completion tokens, one is Gemini 3.7 Flash at 12,284, and two are Claude Fable 5.1 at 65,536. There are also two GPT-6.1 Sol worker outputs with `finish_reason=error`. Nineteen worker outputs have empty text, including both capped final Claude outputs.

The two empty capped final answers are repeat 3, `hle-diamond-66ec2799da20d530acf3bf1e`, and repeat 4, `hle-diamond-670fe86a7e294dc6ad20c1ba`. Both have `error=null`, but the judge correctly assigns false because there is no answer. Thus the summaries report 238 completed workflows, while only 236 have a nonempty final response and no explicit workflow error. The other caps occurred in intermediate workers: a later final response does not mean the workflow had complete intermediate reasoning.

**Handled correctly:** empty final answers stayed in the denominator and were not awarded credit. **Unresolved:** worker caps/errors were allowed into the recorded workflows and are missing from the headline failure counts. These results describe this execution budget and its failure behavior, not unrestricted conductor performance. The current `Runner.step_failure` detects caps, errors, and empty text, and the current workflow benchmark calls `fail_on_incomplete=True`; that safeguard does not retroactively repair these artifacts. Verify it in a fresh run and report worker failures separately. Do not selectively retry only incorrect capped cases: use a declared policy covering every capped case, or a fresh full cohort.

### 2. Direct Fugu answers were truncated

Fugu Max has caps by repeat of **4, 4, 0, 1**. Six capped attempts are image questions and three are text questions. Recorded completion usage clusters around 32,764–32,768 and 65,531–65,532 tokens. All nine lack an extracted FINAL answer; inspected endings cut off mid-analysis, and all were graded incorrect. Some already contain incorrect reasoning, so truncation is not established as the sole cause of failure.

Fugu Ultra v2 has one capped response: repeat 4, `hle-diamond-67b14322ac14e54ff5f30274`, at 98,945 reported completion tokens. Its response is unfinished and graded incorrect. Internal usage aggregation prevents inferring the exact outer request budget from this total.

**Handled correctly:** truncated responses were retained and semantically graded, without dropping hard questions. **Unresolved:** no saved replacement for these capped answers is present. Even if every capped direct answer became correct, the maximum mean@4 increase would be 3.69 percentage points for Max and 0.41 for Ultra; these are bounds, not predicted improvements. A budget sensitivity run should use frozen generation settings and preserve the original cohort.

### 3. Infrastructure failures affect apparent model performance

All original Claude OpenRouter attempts failed with HTTP 402 insufficient credits. The 0% summaries are infrastructure outcomes, not evidence of Claude's reasoning ability. The replacement Anthropic batch cohort has 244/244 nonempty successful generations, all `end_turn`, and no remaining judge errors. Its script uses a 128,000-token budget; the largest saved completion is 126,107. This successfully addresses the failed route, but it changes the route and settings and must be labeled as the replacement cohort.

Astra retains three failures: one empty/error response in the baseline and two HTTP 429 credit-verification admission failures in repeat 2. The three are scored false. This preserves the denominator appropriately for end-to-end reliability, but understates capability conditional on a successful request. Report failures separately and recover infrastructure failures under a consistent policy if the intended claim is model ability.

The conductor retains six HTTP 404 failures saying “No endpoints found that support image input,” across five unique image questions. Images were included, but endpoint capability was incompatible with the selected workflow. Fix capability-aware routing and freeze the revised configuration before evaluating again. Excluding those questions would change the benchmark.

### 4. Historical settings are not fully reproducible from today's files

All four conductor repeats record the same execution configuration hash, `94c59d6b99466abb55d39487b726ee93eb7ae2b1fdfdc9a3120cf66626a9efa4`, and prompt hash, `432bfab01278483d1355c9e7ba271a75294c5b1055b3e8456002cc29eb0ea099`. This supports within-cohort consistency. However, today's prompt has a different hash, and today's frontier configuration contains Claude Opus 5.5 and larger DeepSeek limits, while the saved worker outputs use Claude Fable 5.1 and show 40,960-token DeepSeek caps. Today's configuration cannot be used to interpret the historical output budgets or worker identities.

Direct result summaries do not persist the generation token budget, seed, reasoning effort, or complete request settings. Launch scripts have environment overrides; some saved summaries also describe native batch generation where today's repeat scripts default to standard requests. Rejudging can rewrite mode metadata. Use logs and raw responses as evidence, and save an immutable manifest containing model/provider identities, prompts, dataset revision/hash, budgets, sampling settings, code revision, and all request attempts for future runs. The planner request records already retain considerably more of this provenance.

### 5. Scores are not compute-matched, and judge validation is limited

Direct requests use temperature 0 in the common runner; conductor plans record temperature 0.1 with repeat seeds 42–45, and workflow summaries record worker temperature 0.2. Anthropic batch requests omit temperature. The conductor uses multiple models and calls per question; router models can also perform internal work. These are comparisons of configured systems, not a controlled test isolating orchestration at equal compute or sampling settings. Conductor execution cost summaries also omit planner generation cost; recovered direct runs lack complete retry cost histories. Do not claim equal cost from these summaries.

All cohorts use Gemini 3.8 Flash as judge, with semantic-equivalence instructions and strict verdict validation. This is consistent but does not establish scientific grading accuracy. The judge sees the full question, gold answer, reference rationale, candidate response, and extracted final answer, but does not receive the question image through the shared judge payload. This can be adequate for comparison with explicit gold answers, but image-dependent ambiguities and disputed derivations warrant manual review or an independent judge with image access. Judge finish reasons and token usage are not retained, so a separate judge-cap audit is unavailable; valid saved verdicts establish parse success, not absence of all judge limitations.

### 6. Benchmark scope and statistical interpretation

These results cover **HLE-Diamond Physics**, not all HLE-Diamond subjects. The local question file has no source dataset revision or extraction manifest, so completeness relative to an upstream Physics split and train/test contamination cannot be certified from this checkout.

There are 61 distinct questions, not 244 independent question observations. Mean@4 averages the four outcomes per question; pass@4 means at least one of four answers was correct according to the gold-aware judge. It is an oracle success metric, not demonstrated deployed answer selection. The four samples do not prove statistical independence, especially with temperature 0, and model routing can change across requests. No question has exactly the same nonempty response in all four repeats, which rules out wholesale four-way copying but does not prove independence.

For comparative claims, use paired question-level bootstrap intervals, keeping all four outcomes for each question together. The conductor and Astra both pass 48 questions at pass@4, but that equal count does not imply equal accuracy, cost, or solved question sets. Existing per-run accuracy intervals do not establish significance of differences between systems.

## Common pitfalls checked and handled correctly

| Potential concern | Evidence and handling |
| --- | --- |
| Different or duplicate question sets, omitted image questions | Each of the 32 files contains exactly the same 61 unique IDs. Question, gold answer, and reference rationale match the canonical local file in every record. All 21 image questions have embedded data URLs. All saved conductor plan images match the canonical images. Generation code forwards images to direct solvers, the planner, and workers. Endpoint failures remain disclosed above. |
| Unresolved judge/API parsing failures counted as ordinary wrong answers | No saved `judge_error` remains, and all 1,952 records have boolean verdicts. `correct` agrees with `judge_correct`. Strict parsing rejects duplicate, missing, or unexpected verdict IDs. Aggregation refuses unjudged records or judge errors. Failed generations are deliberately assigned false without a judge response; “judged questions” therefore includes policy-assigned failures. |
| Formatting-only false negatives | The judge receives the entire candidate response and is explicitly instructed to grade the actual answer rather than require `FINAL:`. Missing extracted markers are tracked: Max has 10 nonempty responses without one, including its 9 caps; Ultra has 4, including its cap. This avoids automatically failing every missing marker, while incomplete answers can still legitimately fail. |
| Incorrect mean@4/pass@4 arithmetic or missing repeats | Independently recomputed all seven aggregates from source results; total correct, mean@4, passed-question count, and pass@4 all match. Each has four complete question sets. Astra's baseline is correctly included as repeat 1. |
| Dropping generation failures to inflate the denominator | Saved aggregates keep all 244 attempts per cohort, including Astra and conductor failures. The original failed Claude directories are separate and absent from the successful Anthropic aggregate. |
| Gold/reference answers accidentally passed to solvers | The inspected direct request builders send only question, instruction, and image. Planner construction uses question plus image; worker construction uses question, image, and allowed worker context. Gold and rationale are used for judging and recordkeeping. This addresses harness leakage in the inspected code; the historical prompt is identified only by its stored hash, and training contamination is not established either way. |
| Recovery retries counted as extra independent samples | Each aggregate draws from four run result snapshots, one outcome per question per run. Recovery updates those snapshots; it does not add repeat slots. Anthropic recovery preserves batch IDs and retries pending generations. Four repeat slots remain distinct from up to three infrastructure attempts in a slot. Complete histories for other recovered solver requests are unavailable. |
| Planner output caps or invalid plans | All 244 retained plans end with `stop`, have no plan errors, and their request files contain one saved request record per question. Seeds 42–45 and the 8,192-token planning limit are recorded. There is no observed retained planner truncation. |

## Evidence files and next actions

- [Machine-readable audit evidence](hle_diamond_audit_evidence.json): per-run counts, dataset/result hashes, planner settings, and independently recomputed aggregate checks.
- [Flagged records and worker steps](hle_diamond_audit_flags.csv): exact paths, example IDs, cap/error flags, token usage, final verdicts, and judge reasons. Missing FINAL markers are diagnostic flags, not automatically methodological failures.

Priority: exclude the failed Claude route from capability tables; disclose worker caps and the two falsely “completed” empty finals; repair image routing; freeze historical settings; then run a declared cap/infrastructure recovery or fresh-cohort policy. The existing figures can be reported as observed system performance with these qualifications. They should not be presented as clean, uncapped, compute-matched model capability estimates.
