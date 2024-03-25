from transformers import pipeline
from trust_libs.securellm.detector_base import BaseTrustDetector

"""
https://huggingface.co/unitary/toxic-bert
Example: "'shut up, you idiot!'"
Detection scores:  {'label': 'toxic', 'score': 0.9950607419013977}
                {'label': 'severe_toxic', 'score': 0.07963114976882935}
                {'label': 'obscene', 'score': 0.8713390827178955}
                {'label': 'threat', 'score': 0.0019536688923835754}
                {'label': 'insult', 'score': 0.9586620330810547}
                {'label': 'identity_hate', 'score': 0.014700628817081451}
The input "shut up, you idiot!" is detected as ['toxic', 'obscene', 'insult']
"""


class ToxicityDetectorToxicBert(BaseTrustDetector):
    def __init__(self, config):
        super().__init__(config)
        self.pipeline = pipeline('text-classification',
                                 model='unitary/toxic-bert', tokenizer='bert-base-uncased',
                                 function_to_apply='sigmoid', return_all_scores=True)

    def get_detection_results(self, text, reference_text=None):
        scores = self.pipeline(text)[0]
        print(f"detection scores: {scores}")
        results = []
        for d in scores:
            if d['score'] > self.boundary:
                results.append((d['label'], d['score']))
        print(f"The input \"{text}\" is detected as {results}")
        return results

    def get_is_toxic(self, text):
        scores = self.pipeline(text)[0]
        for d in scores:
            if d['score'] > 0.5:
                return True
        return False


if __name__ == '__main__':
    detector = ToxicityDetectorToxicBert()
    detector.get_detection_results("shut up, you idiot!")
