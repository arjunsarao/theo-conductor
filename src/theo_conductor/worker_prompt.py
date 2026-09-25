"""Render the prompt supplied to every dispatched worker."""

from pathlib import Path
from string import Template


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_PROMPT_PATH = PROJECT_ROOT / "worker-prompt.txt"


def build_worker_prompt(
    *,
    instruction: str,
    question: str,
    context: dict[str, object],
    prompt_path: str | Path = DEFAULT_WORKER_PROMPT_PATH,
) -> str:
    """Render a worker request from the repository-level prompt template."""
    context_blocks = []
    for step_id, output in context.items():
        if step_id == "artifacts":
            context_blocks.append(f"<artifacts>{output}</artifacts>")
        else:
            context_blocks.append(f"<step_output id={step_id}>{output}</step_output>")

    template = Template(Path(prompt_path).read_text(encoding="utf-8"))
    return template.substitute(
        question=question,
        context="\n".join(context_blocks) or "(none)",
        instruction=instruction,
    )
