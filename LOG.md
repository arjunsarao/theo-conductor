# Log

## July 3rd, 2026

Today was generally working on the infrastructure surrounding the system, preparing it for training.

- Implemented json parsing of conductor output.
- Implemented scheduliing via topological sort.
- Added model registry.
- Implemented logging.
- Add dummy models to view testing and data flow through a 

## July 6th, 2026

- Wrote system prompt for conductor
- Implemented prompt configuration
- Added HLE dataset.

## July 7th, 2026

- Added GPQA dataset
- Implemented physics-only filtering for HLE and GPQA
- Refactored model registry to consume yaml files
- Added configs for worker agent pools with small local models, large local models, and frontier models.
- Refactored to use `datasets` library for HLE and GPQA.
- Working on train.py

## July 9th, 2026

- Implemented `grpo.py`
- Implemented `train.py`
- Updated HLE and GPQA to filter for physics-domains as well.

## July 13th, 2026

- Fixed up `main.py`, added functionality to provide question to conductor from main.
- Add tests
- Improve logging

## July 14th, 2026

- Fixed env stuff so now I can launch the vLLM servers for the models.
- Added more workflow examples to the prompt.
- Added workflow viewer.
- Added deterministic MegaScience train/validation splitting with a 2,000-example subset.
- Added validation metrics and periodic evaluation to GRPO training.
- Added Weights & Biases tracking and configuration options.
- Added GRPO preflight checks for dataset size, context length, reward tiers, vLLM execution, and checkpoints.
- Added explicit final-answer extraction and optional workflow execution during reward calculation.
- Added numeric tolerance and symbolic equivalence checks for scientific answers.
