from hallucination_pipeline.wrapper_for_hallucination_pipe import HallucinationPipePostWrapper
from trust_libs.trust_detector import TrustDetector


class PostProcessor:
    def __init__(self):
        self.fedml_security_detector_config = {
            "enable_toxicity": True,
            "enable_gibberish": True,
            "enable_unsafe_prompt": True
        }
        self.error_type = []

    def post_processing(self, raw_output):
        to_reask = False
        logging_info = ""
        new_text = ""
        input_trust_detector = TrustDetector(self.fedml_security_detector_config)
        if input_trust_detector.is_unsafe_content_detected(raw_output):
            logging_info = "unsafe contents detected. "
        post_wrapper = HallucinationPipePostWrapper()
        new_text = post_wrapper.process(raw_output)
        if not self.detect_if_errors_fixed(raw_output, new_text):
            to_reask = True
            logging_info = f"error type: {self.error_type}"
        return new_text, to_reask, logging_info

    def detect_error_type(self): # todo:
        self.error_type = []

    def detect_if_errors_fixed(self, original_text, new_text):  # todo
        return True
