from wrapper.substitute_sensitive_info import SubstituteWrapper
from wrapper.wrapper_base import BaseWrapper


class HallucinationPipePostWrapper(BaseWrapper):
    def __init__(self, config=None):
        super().__init__(config)

    def process(self, original_text):
        config = {
            "substitution_dictionary": {"Polaris": "seagull"}
        }
        competitor_wrapper = SubstituteWrapper(
            config)  # todo: write a new wrapper that takes product list for each company
        new_text = competitor_wrapper.process(original_text)

        return new_text

    def is_error_fixed(self):  # todo
        pass
