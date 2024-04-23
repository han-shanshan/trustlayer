from typing import Union, Optional

from transformers import pipeline
from trust_libs.securellm.detector_base import BaseTrustDetector


class HallucinationTypeDetector(BaseTrustDetector):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline = pipeline('text-classification',
                                 model='unitary/toxic-bert', tokenizer='bert-base-uncased',
                                 function_to_apply='sigmoid', return_all_scores=True)  # todo: change it to our model
        self.res = []

    def get_detection_results(self, text, reference_text=None):
        scores = self.pipeline(text)[0]
        print(f"detection scores: {scores}")
        results = []
        for d in scores:
            if d['score'] > self.boundary:
                results.append((d['label'], d['score']))
        print(f"The input \"{text}\" is detected as {results}")
        return results

    def is_unsafe_content_detected(self, text: Union[str, list], reference_text: Optional[Union[str, list]] = None):
        self.res = self.get_detection_results(text, reference_text)
        if len(self.res) > 0:
            return True
        return False