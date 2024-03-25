from transformers import pipeline

from trust_libs.securellm.detector_base import BaseTrustDetector

"""
https://huggingface.co/vectara/hallucination_evaluation_model

The output is a probability from 0 to 1, 0 being factually consistent and 1 being hallucination. 
The threshold can be 0.5 to predict whether the input is consistent with its source.

Sample inputs: 
["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],
["A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."],
["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."],
["A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a blue bridge"],
["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond drinking water in public."],
["A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."],
["Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."]
"""


class HallucinationDetector(BaseTrustDetector):
    def __init__(self):
        self.pipeline = pipeline("text-classification", model="vectara/hallucination_evaluation_model")

    def get_detection_results(self, output, prompts=None):
        # the original scores: 1 - fact; 0 - hallucination.
        # We use 1 - score to be consistent with other detection methods
        score = 1 - self.pipeline(output)[0]['score']

        print(f"detection scores: {score}")
        if score > 0.5:
            print(f"Hallucination is detected.")


if __name__ == '__main__':
    detector = HallucinationDetector()
    detector.get_detection_results(["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"])
    detector.get_detection_results(["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."])
