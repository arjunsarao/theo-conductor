from theo_conductor.worker_prompt import build_worker_prompt


def test_worker_prompt_is_rendered_from_template(tmp_path):
    template = tmp_path / "worker-prompt.txt"
    template.write_text("Q=$question\nC=$context\nI=$instruction", encoding="utf-8")

    prompt = build_worker_prompt(
        question="Why?",
        context={"solver": "Because."},
        instruction="Check it.",
        prompt_path=template,
    )

    assert prompt == (
        "Q=Why?\n"
        "C=<step_output id=solver>Because.</step_output>\n"
        "I=Check it."
    )

