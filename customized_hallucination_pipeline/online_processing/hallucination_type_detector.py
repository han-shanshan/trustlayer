from typing import Union, Optional

from customized_hallucination_pipeline.test_use_case import HALLUCINATION_INFERENCE_CONFIG
from utils.constants import CUSTOMIZED_HALLUCINATION_TASK_NAME
from inference.inference_engine import InferenceEngine
from safety_detection.securellm.detector_base import BaseTrustDetector


class HallucinationTypeDetector(BaseTrustDetector):
    def __init__(self, config=HALLUCINATION_INFERENCE_CONFIG):
        super().__init__(config)
        self.res = None
        print(f"config = {config}")
        self.inference_engine = InferenceEngine(task_name=CUSTOMIZED_HALLUCINATION_TASK_NAME, config=config,
                                                problem_type="single_label_classification")

    def get_detection_results(self, output, reference_text=None):
        result = self.inference_engine.inference([reference_text, output])
        print(f"result = {result}")
        return result

    def is_unsafe_content_detected(self, text: Union[str, list], reference_text: Optional[Union[str, list]] = None):
        self.res = self.get_detection_results(text, reference_text)
        if self.res == "Normal response":
            return False
        return True

    def get_error_type(self):
        return self.res
