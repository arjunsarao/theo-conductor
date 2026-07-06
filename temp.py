from string import Template
import os


MODELS = ["Kimi k2.6", "DeepSeek v4 Flash", "GLM 5.2", "Qwen3.7 Max"]
TOOLS = ["Python REPL", "Calculator", "Web Search"]
QUERY = "What is the meaning of life?"


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


def main() -> None:
    examples_dir = "examples"
    examples = []
    for filename in os.listdir(examples_dir):
        if filename.endswith(".json"):
            with open(os.path.join(examples_dir, filename), "r") as f:
                examples.append(f.read())
    print(build_prompt(MODELS, TOOLS, examples, QUERY))

if __name__ == "__main__":
    main()
