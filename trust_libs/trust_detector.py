from trust_libs.securellm.toxicity.toxicity_detector_toxic_bert import ToxicityDetectorToxicBert
from typing import Union, Optional


class TrustDetector:
    def __init__(self, config):
        self.detect_results = []
        self.detectors = []
        if 'enable_toxicity' in config and config['enable_toxicity']:
            self.detectors.append(("toxicity", ToxicityDetectorToxicBert(config)))

    def get_detection_results(self):
        return self.detect_results

    def is_unsafe_content_detected(self, text: Union[str, list], reference_text: Optional[Union[str, list]] = None):
        self.detect_results = []
        for detector_type, detector_entity in self.detectors:
            res = detector_entity.get_detection_results(text, reference_text)
            self.detect_results.append((detector_type, res))
        for detector_type, res in self.detect_results:
            if len(res) > 0:
                return True
        return False
