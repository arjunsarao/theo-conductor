from pathlib import Path
from string import Template


def build_prompt(model_list: list[str], tool_list: list[str], example_list: list[str], query: str) -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "conductor-prompt.txt"
    with prompt_path.open() as f:
        prompt_template = f.read()

    template = Template(prompt_template)

    return template.substitute(
        models="\n".join(model_list),
        tools="\n".join(tool_list),
        examples="\n".join(example_list),
        query=query,
    )
