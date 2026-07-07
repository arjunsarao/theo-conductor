Based on the two papers in the repo, I’d treat your “fugu-ultra clone for physics” less like “build a better single physics model” and more like “train a small conductor to assemble physics-solving workflows over specialized workers.”

The repo is already pointed in the right direction: [runner.py](/Users/arjunsarao/Documents/GitHub/theo-conductor/src/theo_conductor/runner.py:1) executes DAG-style workflows, [schema.py](/Users/arjunsarao/Documents/GitHub/theo-conductor/src/theo_conductor/schema.py:1) defines steps/tasks, and `conductor-prompt.txt` already frames the conductor as a JSON workflow generator. The missing pieces are training data, reward/eval, and an RL/SFT loop.

**Recommended Next Steps**

1. **Narrow the target benchmark first**
   
   For physics, do not start with all of HLE. Create a physics-only evaluation slice:
   - HLE physics questions
   - GPQA physics / diamond subset if available
   - MegaScience physics examples
   - maybe graduate textbook-style mechanics, E&M, QM, stat mech, and astro problems

   Your conductor will only get better if the reward signal is crisp. Physics needs exact-answer checks where possible, plus rubric/LLM-judge checks for derivations.

2. **Define your worker pool**
   
   Start small and explicit. For example:
   - `physics_reasoner`: DeepSeek-R1-Distill-Qwen-32B or another strong reasoning model
   - `math_solver`: Gemma/Qwen math-tuned model
   - `coder`: Qwen2.5-Coder for numerical/symbolic checks
   - `verifier`: a separate reasoning model prompted to find mistakes
   - `synthesizer`: best general model you can afford

   Add model metadata to the registry: domain tags, context length, cost, tool support, and whether it is good at derivation, numerical computation, symbolic manipulation, or critique.

3. **Implement the conductor loop end-to-end before training**
   
   Right now you have the worker DAG runner, but you still need:
   - conductor model call: question -> JSON workflow
   - JSON parsing/repair/validation
   - execution through `Runner`
   - final answer extraction
   - reward computation
   - trace logging

   The critical object is a saved trajectory:

   ```json
   {
     "question": "...",
     "gold_answer": "...",
     "workflow": [...],
     "worker_outputs": {...},
     "final_answer": "...",
     "reward": 0.0,
     "cost": 0.034,
     "latency_ms": 42100
   }
   ```

4. **Only then move to GRPO/RL**
   
   The Sakana Fugu report emphasizes learned orchestrators, adaptive scaffolds, evolutionary/RL training, and Fugu-Ultra prioritizing quality on hard tasks. The Conductor paper similarly emphasizes RL discovering coordination strategies over worker pools, including prompt engineering and topologies.

   Your first GRPO reward can be simple:
   - `1.0`: final answer correct
   - `0.5`: reasoning substantially correct but final answer wrong/incomplete
   - `0.0`: wrong
   - `-0.1`: invalid JSON/workflow
   - small penalties for excessive cost/latency

   For physics, I’d also add:
   - units/dimensions reward
   - numerical tolerance reward
   - symbolic equivalence where possible
   - penalty for unsupported shortcuts or missing assumptions

7. **Fix a few repo issues before training**
   
   I’d prioritize these implementation chores:
   - make `validate.py` real: schema validation, model existence, final-step check, acyclic workflow
   - make `Runner` call `model_registry.validate_task(task)` before execution
   - fix test mismatch: `ModelRegistry.get` currently raises `"Model '999' not found"`, but the test expects `"Unknown model_idx"`
   - fix `main.py`: `async_main()` is never awaited
   - add a conductor client that turns `build_prompt(...)` output into a `Task`
   - improve context serialization in `openai_compat.py`; it currently inserts `StepOutput` objects directly into XML-ish blocks

8. **Build the physics grader early**
   
   This is the highest-leverage part. A Fugu-Ultra clone for physics will live or die by reward quality. Implement graders in layers:
   - exact string / multiple choice
   - numeric tolerance
   - unit-aware comparison
   - symbolic equivalence via SymPy
   - LLM judge for derivation quality
   - optional verifier-model disagreement signal

**My Suggested Milestone Order**

1. Physics benchmark loader: HLE/MegaScience filtered to physics.
2. Worker registry with 3-5 real models.
3. End-to-end conductor -> workflow -> runner -> grader trace.
4. Baselines: single best worker, best-of-N, parallel adjudicator.
5. SFT router from per-worker physics success rates.
6. Template workflow distillation.
7. GRPO over conductor JSON workflows.
8. Add cost/latency-aware reward once quality improves.

One subtle but important recommendation: do not clone Fugu-Ultra’s full generality first. Clone the *training shape*: learned orchestration over heterogeneous workers with end-to-end reward. Keep the physics domain narrow enough that your reward is trustworthy.

Sources checked: Sakana Fugu Technical Report, arXiv `2606.21228`; Learning to Orchestrate Agents in Natural Language with the Conductor, arXiv `2512.04388`.