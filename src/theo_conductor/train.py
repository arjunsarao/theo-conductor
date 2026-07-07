import os
import random
from typing import Literal

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForMultimodalLM

load_dotenv()


def format_gpqa_question(ex) -> tuple[str, Literal["A", "B", "C", "D"]]:
    question = ex["Question"]
    correct_answer = ex["Correct Answer"]
    incorrect_answers = ex["Incorrect Answer"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]
    mc = [correct_answer] + list(incorrect_answers)
    random.shuffle(mc)
    correct_letter = ["A", "B", "C", "D"][mc.index(correct_answer)]
    question_str = f"Question: {question}\n"
    for i, answer in enumerate(mc):
        question_str += f"{['A', 'B', 'C', 'D'][i]}. {answer}\n"
    return question_str.strip(), correct_letter


hle = load_dataset("cais/hle", split="test", token=os.getenv("HF_TOKEN"))
hle_physics = hle.filter(lambda ex: ex["category"] == "Physics")

gpqa = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train", token=os.getenv("HF_TOKEN"))
gpqa_physics = gpqa.filter(lambda ex: ex.get("High-level domain") == "Physics")

# For hle, i just need the question and answer columns, but for GPQA, i need Question, Correct Answer, and the Incorrect Answers columns. I will merge the two datasets and keep just the question and answer columns.

# Load Qwen3.5-9B
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B-Base")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-9B-Base")
messages = [
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
inputs = processor.apply_chat_template(
    messages,
    chat_template=processor.tokenizer.chat_template,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
