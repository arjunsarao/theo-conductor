from theo_conductor.train import resolve_chat_template


class DummyTokenizer:
    def __init__(self, template: str | None = None):
        self.chat_template = template


class DummyProcessor:
    def __init__(self, template: str | None = None, tokenizer_template: str | None = None):
        self.chat_template = template
        self.tokenizer = DummyTokenizer(tokenizer_template)


def test_resolve_chat_template_falls_back_to_tokenizer_template():
    processor = DummyProcessor(tokenizer_template="tokenizer-template")

    assert resolve_chat_template(processor) == "tokenizer-template"


def test_resolve_chat_template_prefers_processor_template():
    processor = DummyProcessor(template="processor-template", tokenizer_template="tokenizer-template")

    assert resolve_chat_template(processor) == "processor-template"
