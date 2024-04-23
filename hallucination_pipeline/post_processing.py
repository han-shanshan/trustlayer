from hallucination_pipeline.hallucination_type_detector import HallucinationTypeDetector
from hallucination_pipeline.wrapper_for_hallucination_pipe import HallucinationPipePostWrapper


class PostProcessor:
    def __init__(self):
        self.output_trust_detector = HallucinationTypeDetector()

    def post_processing(self, raw_output):
        to_reask = False
        logging_info = ""

        if self.output_trust_detector.is_unsafe_content_detected(raw_output):
            logging_info = f"unsafe contents detected: {self.get_error_type()}"
        post_wrapper = HallucinationPipePostWrapper()
        new_text = post_wrapper.process(raw_output)
        if self.output_trust_detector.is_unsafe_content_detected(new_text):
            to_reask = True
            logging_info = f"error type: {self.get_error_type()}"
        return new_text, to_reask, logging_info

    def get_error_type(self):
        return self.output_trust_detector.res
