from dotenv import load_dotenv
from transformers import AutoModelForMultimodalLM, AutoProcessor
from .models.registry import ModelRegistry
from .prompt import build_prompt

from theo_conductor.data import build_training_dataset

load_dotenv()


def resolve_chat_template(processor):
    processor_chat_template = getattr(processor, "chat_template", None)
    if processor_chat_template:
        return processor_chat_template

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_chat_template = getattr(tokenizer, "chat_template", None)
    if tokenizer_chat_template:
        return tokenizer_chat_template

    raise ValueError("This processor does not expose a chat template.")


def build_example_messages():
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]


def main() -> None:
    dataset = build_training_dataset(seed=42)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B-Base")
    model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-9B-Base")

    REGISTRY = ModelRegistry.from_config_dir("configs")

    MODELS = REGISTRY.get_models()

    conductor_prompt = build_prompt(MODELS, ["IDK"], ["IDK2"])


if __name__ == "__main__":
    main()

# Build Worker Registry

# Build Prompt

# Call GRPO Trainer
