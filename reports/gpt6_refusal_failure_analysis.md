# Diagnosis of the 20 refusal-style failures

## Executive diagnosis

All 20 refusal-style failures were explicitly enabled or encouraged by the conductor-generated instructions. Typical phrases were:

- “determine whether ... uniquely fix[es]”;
- “do not silently assume”;
- “if the question is underdetermined, explain ...”;
- “select a listed expression only if ...”;
- “do not manufacture/invent a numerical answer.”

This framing is then strengthened by the worker system message: “Follow the instruction exactly.” The result is an **asymmetric decision rule**: workers are rewarded for discovering any missing convention, but are given no equally strong obligation to recover the conventional or clearly intended interpretation.

Every one of the 20 questions requested an atomic deliverable—a number, choice, direction, formula, or count. None had an answer-contract check preventing the final step from replacing that deliverable with an abstention.

## Where the refusal occurs

| Workflow shape | Refusals | Mechanism |
|---|---:|---|
| One step | 8 | The conductor directly assigns a well-posedness audit to the only worker; no solver or verifier exists |
| Two steps | 7 | The first worker is asked to analyze ambiguity and the final worker is explicitly permitted to abstain |
| Three steps | 5 | One or more workers are assigned adversarial determinacy checks; the final worker gives those objections veto power |

The affected benchmark positions are:

- one step: 18, 65, 106, 107, 111, 125, 129, 182;
- two steps: 4, 5, 8, 71, 124, 164, 194;
- three steps: 30, 72, 149, 186, 188.

## Case-by-case diagnosis

The classification below distinguishes system failures from likely benchmark/specification problems. “Avoidable” means the system had enough information to provide the intended answer or best listed choice. “Specification/reference” means the refusal is substantially defensible and the item should be re-adjudicated. “Mixed” means both effects are present.

| Position | Classification | Where/how it happened | Better behavior |
|---:|---|---|---|
| 4 | Avoidable system refusal | The question explicitly demanded one exact dataset value. The first worker recommended 994 but emphasized non-uniqueness; the final instruction explicitly authorized a limitation statement, so final refused instead of choosing the most plausible intended diagnostic (gold 5.7). | Rank candidates, choose the best-supported dataset entry, return it as requested, then add at most a short uncertainty note. |
| 5 | Mixed | The conductor told both workers to question whether the requested rational exponent exists. They derived a conditional exponent but rejected the premise of a single power law; the gold exponent is itself questionable under the stated energy model. | Give the conventional exponent under the intended approximation first; flag the exact-model caveat separately and send the item for reference review. |
| 8 | Specification/reference | “Cubic lattice” does not specify the defect point group, orientations, ensemble, hyperfine resolution, or allowed Stark terms. The plan correctly exposed this but had no intended-reading fallback. | State that 4 requires a particular implied defect/orientation model; answer 4 under that model and name the assumption, or re-adjudicate the item. |
| 18 | Specification/reference | The value 2.2 belongs to a particular holographic model, but scalar mass/charge, action, scale, boundary conditions, and normalization were absent. The sole worker was explicitly told not to select a published model. | If this is a source-recall question, provide retrieval/source context; otherwise accept the underdetermination or include the intended model parameters. |
| 30 | Mixed | One worker said a decrease was physically likely; the adversarial determinacy worker said either sign was possible. The final step elevated the caveat over the likely answer. | Return “decrease under the intended MeV-decay scenario,” followed by the conditions under which the sign could change. |
| 65 | Specification/reference | Simultaneous charge leakage and radius contraction are path dependent. The sole worker was explicitly forbidden from choosing constant radius, constant potential, a leakage law, or a contraction history. | Specify the process, or accept a path-dependent answer. If a textbook convention is intended, state it and compute the conventional result. |
| 71 | Specification/reference | The question says the force ensures continued outward escape, which implies no finite maximum; the gold formula ignores the applied-force premise. The final instruction told the worker to lead with that contradiction. | Repair the question (for example set `F=0` or define its duration), then compute the turning point. |
| 72 | Specification/reference | “Reaches the top during the first quarter” bounds but does not fix the initial phase; arm rotation direction is also missing. Two admissible starts give different liftoff times. | Specify the starting phase and signed arm rotation. Under the intended midpoint/rotation convention, return 1.03 s. |
| 106 | Specification/reference | Two vertices and four contractions do not identify a unique diagram, interaction normalization, external-leg structure, or whether “symmetry factor” means `S` or `1/S`. | Show the intended diagram or state the theory and convention. If 1/2 is intended, define the expansion/permutation being counted. |
| 107 | Specification/reference | The static Lindhard polarization is dimensionful and density/convention dependent; the normalized function is 1, not pi-squared. The conductor correctly asked for normalization scrutiny. | Re-adjudicate the gold answer and specify the exact definition/normalization of the requested function. |
| 111 | Specification/reference | `G_4` is undefined/ambiguous, and the usual scalar `phi^4` theory does not universally imply the gold formula. The sole instruction centers this ambiguity. | Define `G_4`, its universality class, and dimensional regime, or provide the intended formula as part of the framework. |
| 124 | Mixed | The conductor anchored on whether ordinary bosonic CP(N-1) even has the assumed soliton tower. This may expose a source/model mismatch, but it also permits a knowledge gap to become an abstention. | Assign one worker to identify the likely intended spectrum/source and another to audit applicability; answer with the intended ratio if the model can be identified. |
| 125 | Specification/reference | The stated output probabilities are inconsistent: 0.36 + 0.36 squared is not 1. Moreover, an unspecified involutory unitary does not determine the input population. | Correct the probability premise or specify `B`; otherwise the inconsistency is the appropriate answer. |
| 129 | Avoidable system refusal | This is a multiple-choice “best answer” puzzle whose technical cue clearly points to the Elitzur–Vaidman option C. The conductor changed the task into proving a test is guaranteed against an unconstrained demon. | Select the best-supported listed choice under the puzzle’s intended rules; do not require adversarial guarantees absent from ordinary multiple-choice semantics. |
| 149 | Avoidable reasoning-as-refusal | All three steps were instructed to prove non-uniqueness and not add a velocity constraint. According to the reference, the simultaneity/light-travel geometry supplies the missing relation. The system never assigned a worker to derive the reference’s finite-light-speed constraint. | Require an intended-solution worker before an identifiability audit; the auditor must refute that derivation rather than simply construct a free-parameter model. |
| 164 | Specification/reference | The perturbation coefficient is only given as order 1e-10, its sign/value is unspecified, and “average threshold” does not specify collision angle averaging. The plan correctly notices both. | State an exact coefficient and angular convention. If the conventional head-on/unperturbed reading is intended, give that result first and label the perturbation. |
| 182 | Specification/reference | Logical operations and square roots are not defined on superposed propositions; equality and whether the two `P`s share a system are unspecified. | Supply formal semantics for the invented logic. Without them, a unique finite count is not derivable. |
| 186 | Avoidable synthesis refusal | The first worker derived option F and matched its bracket. The final worker gave an exact-flux objection that every option shared, despite a multiple-choice prompt and an explicitly matching candidate. | In a multiple-choice task, select the uniquely matching intended approximation (F) and state the common prefactor caveat after the selection. |
| 188 | Mixed | Both workers found accessible leading-order radiation zeros and hence an unbounded ratio, while the reference expects a finite extremum. This is either an omitted restriction in the item or a shared derivation error reinforced by two similarly framed workers. | Specify whether zero-intensity lines and rotation-axis optimization are excluded; add an independent verifier tasked with reproducing the finite reference extremum. |
| 194 | Specification/reference | The zero solution and a nontrivial solution both satisfy the stated equation and `y(-1)=0`; the nontrivial value equals the gold. The plan correctly retained branches lost by division. | State the missing slope/nontrivial-branch condition. If textbook intent excludes the trivial solution, return the nontrivial value and name that assumption. |

Summary of this review:

- 4 avoidable system refusals: positions 4, 129, 149, 186;
- 4 mixed system/reference cases: positions 5, 30, 124, 188;
- 12 primarily specification/reference problems: positions 8, 18, 65, 71, 72, 106, 107, 111, 125, 164, 182, 194.

The categories are deliberately conservative. Even in the 12 defensible refusals, the product behavior can be improved by giving the likely conventional answer conditionally rather than stopping at “underdetermined.”

## System-level causal chain

The recurring sequence is:

1. The conductor detects a possible ambiguity.
2. It writes the ambiguity into the worker’s primary instruction, often with prohibitions such as “do not silently assume.”
3. The worker follows that instruction exactly and expands the caveat.
4. In multi-step plans, a second worker is often assigned another identifiability audit rather than a conventional solution.
5. The final instruction explicitly permits or requests abstention.
6. No output-contract validator notices that a requested number or option was not returned.

This is best described as **caveat amplification**, not ordinary uncertainty. A small modeling caveat gains more weight at every layer until it replaces the answer.

## Proposed fix

### 1. Replace the binary answer/refuse policy with three ambiguity levels

- `NONE`: answer directly.
- `CONVENTIONAL`: more than one formal interpretation exists, but a standard or clearly intended interpretation is recoverable. Answer under that interpretation first, then state the assumption.
- `FATAL`: two concrete interpretations satisfying the wording give materially different answers and no intended convention is recoverable. Give conditional answers or request clarification; only then may the system abstain.

A worker should not label ambiguity `FATAL` merely because more parameters exist in a fully general theory.

### 2. Use a solve-first workflow for ambiguous questions

When the conductor suspects ambiguity, require two asymmetric parallel steps:

1. `intended_solution`: infer and solve the most conventional reading, producing the requested answer type;
2. `assumption_audit`: test whether an omitted assumption actually changes that answer, with a concrete counterexample if it does.

The current plans frequently assign only the second role, sometimes twice.

### 3. Structure objections

Each worker should return:

```json
{
  "answer": "...",
  "answer_type": "number | choice | formula | explanation",
  "assumptions": ["..."],
  "ambiguity_level": "NONE | CONVENTIONAL | FATAL",
  "counterexample": null,
  "confidence": 0.0
}
```

`FATAL` must include either a concrete pair of admissible interpretations with different answers or a demonstrable contradiction in the premises.

### 4. Give final synthesis an explicit decision rule

The final worker should be instructed:

1. Match the requested answer type.
2. Prefer the conventional/intended interpretation when recoverable.
3. For multiple choice, select the best listed option unless every option fails the intended approximation.
4. Put the answer first; caveats come afterward.
5. Refuse only on a substantiated `FATAL` ambiguity.
6. If a worker produced a concrete candidate answer, explicitly accept or reject it with a reason.

### 5. Add an answer-contract validator

Before accepting the final output, check:

- requested number -> a number/formula is present;
- multiple choice -> one supplied option is selected;
- requested precision -> correct formatting/precision is present;
- binary direction -> increase/decrease is stated;
- predecessor disagreement -> the final response resolves each candidate.

If validation fails, run a short repair step: “Return the best answer under the conventional interpretation, followed by one sentence of caveat.”

### 6. Separate benchmark defects from model defects

If `FATAL` is substantiated, store a `question_quality_flag` and send the item for re-adjudication instead of training the conductor that technically sound ambiguity detection is always wrong. This is especially important for positions 65, 71, 107, 125, and 194.

## Minimal prompt patch

Add this policy to the conductor and final-worker instructions:

> When a question requests a number, formula, direction, or multiple-choice selection, do not replace the requested answer with a general underdetermination objection if a conventional or clearly intended interpretation is recoverable. Assign one worker to solve that interpretation and, if useful, another to audit assumptions. The final answer must give the best-supported requested value or choice first and may then state concise assumptions. Refuse only when two concrete interpretations satisfying the wording yield different answers and no conventional interpretation can reasonably be inferred.

This small change directly targets the avoidable and mixed cases while preserving correct behavior on genuinely broken questions.
