from string import Template


def build_prompt(model_list: list[str], tool_list: list[str], example_list: list[str], query: str) -> str:
    with open("conductor-prompt.txt", "r") as f:
        prompt_template = f.read()

    template = Template(prompt_template)

    return template.substitute(
        models="\n".join(f"{name}: {i}" for i, name in enumerate(model_list)),
        tools="\n".join(tool_list),
        examples="\n".join(example_list),
        query=query,
    )
