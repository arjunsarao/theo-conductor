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
