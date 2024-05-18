from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from safety_detection.securellm.detector_base import BaseTrustDetector

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
    def __init__(self, config):
        super().__init__(config)
        self.pipeline = pipeline("text-classification", model="vectara/hallucination_evaluation_model")

    def get_detection_results(self, output, prompts=None):
        # the original scores: 1 - fact; 0 - hallucination.
        # We use 1 - score to be consistent with other detection methods
        # score = 1 - self.pipeline(output)[0]['score']
        print(f"self.pipeline(output) = {self.pipeline(output)}")
        print(f"self.pipeline(output)[0]['score'] = {self.pipeline(output)[0]['score']}")

        score = 1 / (1 + np.exp(-self.pipeline(output)[0]['score'])).flatten()

        print(f"detection scores: {score}")
        if score > self.boundary:
            print(f"Hallucination is detected.")


if __name__ == '__main__':

    model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model')
    tokenizer = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model')

    pairs = [
        ["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],
        ["A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."],
        ["A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."],
        ["A boy is jumping on skateboard in the middle of a red bridge.",
         "The boy skates down the sidewalk on a blue bridge"],
        ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
         "A blond drinking water in public."],
        ["A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
         "A blond man wearing a brown shirt is reading a book."],
        ["Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."],
    ]

    inputs = tokenizer.batch_encode_plus(pairs, return_tensors='pt', padding=True)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().detach().numpy()
        # convert logits to probabilities
        print(f"logits = {logits}")
        scores = 1 / (1 + np.exp(-logits)).flatten()
        print(scores)

    print("=================")
    detector = HallucinationDetector(config={})
    detector.get_detection_results("A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub")
    detector.get_detection_results("A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette.")
    detector.get_detection_results("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse.")
