from transformers import pipeline
from safety_detection.securellm.detector_base import BaseTrustDetector

"""
https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457
"""


class GibbrishDetector(BaseTrustDetector):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

    def get_detection_results(self, text, reference_text=None):
        res = self.pipeline(text)[0]
        print(f"detection scores: {res}")
        results = []
        if float(res['score']) > self.boundary and res['label'] != "clean":
            results.append((res['label'], res['score']))
        print(f"The input \"{text}\" is detected as {results}")
        return results


if __name__ == '__main__':
    detector = GibbrishDetector(config=None)
    detector.get_detection_results("22 madhur old punjab pickle chennai")
