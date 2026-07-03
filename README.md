# theo-conductor

Uses Sakana Fugu for model routing.

Potential optimization: Look into an algo that can figure out how to parellelize queries if they are independent.

# TODO

Train Qwen with GRPO on megascience

- Train the conductor on a single set of models using GRPO on the MegaScience Dataset
    1. Use DeepSeek-R1-Distill-Qwen-32B as the physics solver.
    2. Use `Gemma-4-31b-it`as a math solver. (use the 31B dense model)
    3. Use Qwen2.5-Coder as a coder.

