1. **Implement the conductor loop end-to-end before training**
   
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

2. **Only then move to GRPO/RL**
   
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

3. **Fix a few repo issues before training**
   
   I’d prioritize these implementation chores:
   - make `validate.py` real: schema validation, model existence, final-step check, acyclic workflow
   - make `Runner` call `model_registry.validate_task(task)` before execution
   - fix test mismatch: `ModelRegistry.get` currently raises `"Model '999' not found"`, but the test expects `"Unknown model_idx"`
   - fix `main.py`: `async_main()` is never awaited
   - add a conductor client that turns `build_prompt(...)` output into a `Task`
   - improve context serialization in `openai_compat.py`; it currently inserts `StepOutput` objects directly into XML-ish blocks

4. **Build the physics grader early**
   
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
