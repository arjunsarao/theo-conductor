# theo-conductor

Uses Sakana Fugu for model routing.

# TODO

- Train the conductor on a single set of models using GRPO on the MegaScience Dataset
    1. Use DeepSeek-R1-Distill-Qwen-32B as the physics solver.
    2. Use `Gemma-4-31b-it`as a math solver. (use the 31B dense model)
    3. Use Qwen2.5-Coder as a coder.
- Create custom system prompts for conductor and ensemble models.
- Implement runner/dispatch for various llms (make it configurable)
- Add HLE benchmarking from huggingface.
- Question field is added by me, not echoed by the LLM
- Qwen's context length is >200k tokens.

Final filtered data is 1830 questions.

Total calls = 1830 questions * 5 steps per workflow * 4 rollouts
