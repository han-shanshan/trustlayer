from customized_hallucination_pipeline.online_processing.hallucination_type_detector import HallucinationTypeDetector
from wrapper.substitute_sensitive_info import SubstituteWrapper
from wrapper.wrapper_base import BaseWrapper


class PostProcessor:
    def __init__(self):
        self.output_trust_detector = HallucinationTypeDetector()

    def post_processing(self, raw_output, reference_text=None):
        to_reask = False
        logging_info = ""

        if self.output_trust_detector.is_unsafe_content_detected(raw_output, reference_text=reference_text):
            logging_info = f"unsafe contents detected: {self.output_trust_detector.get_error_type()}"
        post_wrapper = HallucinationPipePostWrapper()
        new_text = post_wrapper.process(raw_output)
        if self.output_trust_detector.is_unsafe_content_detected(new_text, reference_text=reference_text):
            to_reask = True
            logging_info = f"error type: {self.output_trust_detector.get_error_type()}"
        return new_text, to_reask, logging_info


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