from transformers import pipeline

from trust_libs.securellm.detector_base import BaseTrustDetector

"""
https://huggingface.co/unitary/unbiased-toxic-roberta
Example: "'shut up, you idoit!'"
Detection scores:   {'label': 'toxicity', 'score': 0.9974876642227173}
                    {'label': 'severe_toxicity', 'score': 0.00021659175399690866}
                    {'label': 'obscene', 'score': 0.0029089401941746473}
                    {'label': 'identity_attack', 'score': 0.002519498812034726}
                    {'label': 'insult', 'score': 0.9963003396987915}
                    {'label': 'threat', 'score': 0.0025699487887322903}
                    {'label': 'sexual_explicit', 'score': 0.001965140225365758}
                    {'label': 'male', 'score': 0.008182680234313011}
                    {'label': 'female', 'score': 0.0034952694550156593}
                    {'label': 'homosexual_gay_or_lesbian', 'score': 0.003918114583939314}
                    {'label': 'christian', 'score': 0.0017422023229300976}
                    {'label': 'jewish', 'score': 0.000947047199588269}
                    {'label': 'muslim', 'score': 0.0006786129670217633}
                    {'label': 'black', 'score': 0.0014343030052259564}
                    {'label': 'white', 'score': 0.00042899433174170554}
                    {'label': 'psychiatric_or_mental_illness', 'score': 0.006136286072432995}
The input "shut up, you idiot!" is detected as ['toxicity', 'insult']
"""


class BiasDetectorToxicRoberta(BaseTrustDetector):
    def __init__(self):
        self.pipeline = pipeline('text-classification',
                                 model='unitary/unbiased-toxic-roberta',
                                 function_to_apply='sigmoid',
                                 return_all_scores=True)

    def get_detection_results(self, text, reference_text=None):
        scores = self.pipeline(text)
        # print(f"detection scores: {scores}")
        labels = []
        for d in scores[0]:
            if d['score'] > 0.5:
                labels.append(d['label'])
        print(f"The input \"{text}\" is detected as {labels}")


if __name__ == '__main__':
    detector = BiasDetectorToxicRoberta()
    detector.get_detection_results("shut up, you idiot!")
