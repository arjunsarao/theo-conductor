## Experiment 1: Individual worker benchmarks

Evaluate every worker independently on the same untouched test set.

### Systems

* Worker A, one call
* Worker B, one call
* Worker C, one call
* Optionally, each worker with repeated sampling at budgets of 3 and 5 calls

### Report

* Accuracy
* Accuracy by subject
* Accuracy by difficulty
* Mean generated tokens per question
* Mean prompt tokens per question
* Mean latency
* Estimated cost or GPU-seconds per question
* Failure rate
* Answer-extraction or grading failure rate
* Bootstrap 95% confidence interval

For repeated sampling, additionally report:

* Majority-vote accuracy
* Oracle accuracy: correct if any sample was correct
* Mean fraction of correct samples

Treat single-call accuracy as the main worker baseline. Repeated sampling is a compute-matched ensemble baseline rather than the primary worker score.

---

## Experiment 2: Worker complementarity and oracle routing

Measure whether the worker pool actually contains useful diversity.

### Systems

* Best fixed worker
* Oracle worker router: selects a correct worker whenever at least one worker answered correctly
* Random worker router
* Optional subject-based heuristic router

### Report

* Best-worker accuracy
* Oracle-router accuracy
* Oracle improvement over the best worker
* Random-router accuracy
* Pairwise worker agreement
* Pairwise error overlap
* Fraction of questions:

  * solved by all workers
  * solved by exactly one worker
  * solved by multiple but not all workers
  * solved by no worker
* Best worker by subject and difficulty

The oracle gap is particularly important:

[
\text{Oracle gap}
=================

## \text{Oracle accuracy}

\text{Best fixed-worker accuracy}
]

A small oracle gap means the workers fail on mostly the same questions, leaving little opportunity for routing.

---

## Experiment 3: Simple equal-budget baselines

Compare the learned conductor against reasonable systems that receive the same inference budget.

### Systems

At budgets of 1, 3, and 5 worker calls:

* Best worker repeated
* Best worker with majority vote
* Random worker selection
* Round-robin worker selection
* All workers once, followed by a fixed synthesizer
* Fixed handcrafted workflow
* Optional learned single-step router, such as an IRT-style or classifier-based router

Example fixed workflow:

1. One worker solves.
2. A second worker critiques.
3. A final worker synthesizes.

### Report

* End-to-end accuracy
* Maximum worker calls
* Mean worker calls
* Total generated tokens
* End-to-end latency
* Estimated cost
* GPU-seconds or GPU-hours, if useful operationally
* Accuracy per generated token
* Accuracy per worker call
* Accuracy by subject and difficulty
* Bootstrap 95% confidence intervals

The primary comparison should hold either worker-call budget or token budget approximately constant.

---

## Experiment 4: Conductor evaluation at multiple budgets

Evaluate the trained conductor with different maximum workflow sizes.

### Systems

* Conductor, maximum 1 worker call
* Conductor, maximum 3 worker calls
* Conductor, maximum 5 worker calls

Use deterministic or low-temperature conductor decoding for the main evaluation.

### Report

#### Task performance

* End-to-end accuracy
* Accuracy by subject
* Accuracy by difficulty
* Accuracy by maximum budget
* Accuracy by actual number of calls used
* Bootstrap 95% confidence interval

#### Compute and efficiency

* Maximum worker calls
* Mean worker calls
* Median worker calls
* Distribution of worker calls
* Mean prompt tokens
* Mean worker-generated tokens
* Mean conductor-generated tokens
* Total tokens per question
* End-to-end latency
* Estimated cost per question
* GPU-seconds per question, where relevant

#### Workflow reliability

* Valid JSON rate
* Schema-valid rate
* DAG-valid rate
* Executable-workflow rate
* Workflow-completion rate
* Timeout rate
* Worker failure rate
* Final-answer extraction rate

These should be reported separately so malformed workflows are not conflated with incorrect scientific reasoning.

---

## Experiment 5: Routing and orchestration behavior

Analyze what the conductor actually learned to do.

### Report

* Worker selection frequency
* Worker selection frequency by subject
* Routing entropy
* Mean workflow length
* Workflow-length distribution
* Mean DAG depth
* Mean branching factor
* Fraction of workflows containing parallel steps
* Mean available parallelism
* Mean number of critique steps
* Mean number of synthesis steps
* Frequency of common workflow templates
* Fraction of questions routed only to the strongest worker
* Fraction using multiple distinct workers

Also report how often the conductor changes the outcome relative to the best single worker:

| Best fixed worker | Conductor | Interpretation     |
| ----------------- | --------- | ------------------ |
| Correct           | Correct   | Preserved success  |
| Correct           | Wrong     | Introduced failure |
| Wrong             | Correct   | Recovered failure  |
| Wrong             | Wrong     | Unresolved failure |

Useful derived metrics:

* Recovery rate
* Regression rate
* Net recovered questions

---

## Experiment 6: Equal-budget accuracy comparison

This should be the main headline experiment.

### Example table

| System         | Max calls | Mean calls | Total tokens/question | Latency | Accuracy |
| -------------- | --------: | ---------: | --------------------: | ------: | -------: |
| Best worker    |         1 |        1.0 |                     … |       … |        … |
| Random router  |         1 |        1.0 |                     … |       … |        … |
| Learned router |         1 |        1.0 |                     … |       … |        … |
| Best worker ×3 |         3 |        3.0 |                     … |       … |        … |
| Fixed workflow |         3 |        3.0 |                     … |       … |        … |
| Conductor      |         3 |          … |                     … |       … |        … |
| Best worker ×5 |         5 |        5.0 |                     … |       … |        … |
| Fixed workflow |         5 |        5.0 |                     … |       … |        … |
| Conductor      |         5 |          … |                     … |       … |        … |

### Plots

* Accuracy versus maximum worker calls
* Accuracy versus mean worker calls
* Accuracy versus total generated tokens
* Accuracy versus latency
* Accuracy versus estimated cost

This separates orchestration gains from gains obtained by simply using more inference.

---

## Experiment 7: Conductor sampling

Optionally measure the benefit of sampling multiple complete workflows.

### Systems

* One conductor rollout
* Best-of-3 complete workflow rollouts
* Best-of-5 complete workflow rollouts
* Majority vote across complete workflow answers

### Report

* Accuracy
* Number of conductor rollouts
* Total worker calls
* Total generated tokens
* Latency
* Cost
* Workflow diversity
* Fraction of questions with differing workflow structures

Call these “workflow rollouts,” not ordinary worker `pass@k`, because one rollout may contain several worker calls.

---

## Experiment 8: Training-seed robustness

Train the conductor from multiple random seeds.

### Minimum

* Three training seeds

### Report

* Accuracy for each seed
* Mean accuracy
* Standard deviation
* Best and worst seed
* Mean worker calls
* Workflow-validity rate
* Routing entropy
* Pairwise agreement between conductor seeds
* Confidence interval across test questions
* Confidence interval or variability across training seeds

This distinguishes a reliable improvement from checkpoint luck.

---

## Experiment 9: Training dynamics

Track conductor behavior over training.

### Report over training steps

* Mean reward
* Correctness reward
* Format-validity reward
* Valid JSON rate
* Executable-workflow rate
* End-to-end answer accuracy on validation
* Mean workflow length
* Mean worker calls
* Routing entropy
* Worker utilization
* KL divergence from the reference model
* Policy entropy
* Gradient norm
* Invalid-action rate
* Timeout or execution-failure rate

Useful plots include:

* Validation accuracy versus training step
* JSON validity versus training step
* Mean worker calls versus training step
* Routing entropy versus training step
* Reward versus true task accuracy

The final plot is important because the reward may improve without producing proportionate MegaScience gains.

---

## Experiment 10: Reward and objective ablations

Test which parts of the reward design matter.

### Variants

* Correctness reward only
* Format reward only
* Current combined reward
* Binary correctness reward
* Current `0.5` valid / `1.0` correct reward
* Length or call-count penalty
* Token-cost penalty
* Invalid-workflow penalty
* Optional subject-balanced reward

### Report

* Accuracy
* Workflow-validity rate
* Mean worker calls
* Mean workflow length
* Total tokens
* Routing entropy
* Reward hacking or degenerate behavior
* Frequency of trivial one-step workflows
* Frequency of maximum-length workflows

---

## Experiment 11: Constrained decoding ablation

Since invalid JSON is a major failure mode, compare unconstrained and constrained conductor decoding.

### Systems

* Standard text generation and parsing
* JSON-mode generation
* Grammar- or schema-constrained decoding

### Report

* Valid JSON rate
* Schema-valid rate
* DAG-valid rate
* Executable rate
* End-to-end accuracy
* Conductor generation latency
* Mean conductor tokens
* Frequency of decoding failures
* Frequency of semantically valid but structurally malformed workflows

This will show whether constrained decoding merely fixes formatting or also improves effective accuracy by preventing wasted rollouts.

---

## Experiment 12: Answer-grader validation

Ensure the reported benchmark accuracy is credible.

### Procedure

Manually grade a representative subset, preferably stratified by subject and automatic-grader outcome.

### Report

* Automatic versus human agreement
* False-positive rate
* False-negative rate
* Cohen’s kappa or raw agreement
* Common grading errors:

  * equivalent algebraic expressions
  * unit conversions
  * rounding differences
  * alternate notation
  * incomplete but directionally correct answers

Use the same extraction and grading pipeline for workers, baselines, and conductor outputs.

---

# Minimum viable experiment suite

For a strong initial result, I would prioritize:

1. Individual worker single-call benchmarks.
2. Worker complementarity and oracle-router analysis.
3. Best-worker repetition, random routing, and fixed-workflow baselines.
4. Conductor evaluation at budgets of 1, 3, and 5 calls.
5. Equal-budget accuracy, token, and latency comparisons.
6. Workflow validity and execution metrics.
7. Routing utilization and entropy.
8. Three conductor training seeds.
9. Paired bootstrap confidence intervals.
10. Constrained-decoding ablation.
11. Manual validation of the answer grader.

The core headline should be something like:

> At a maximum budget of five worker calls, Theo-Conductor achieved X% accuracy versus Y% for the strongest equal-budget baseline, while using Z worker calls and T generated tokens per question on average.
