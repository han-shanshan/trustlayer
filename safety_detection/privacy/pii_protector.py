from transformers import pipeline
import string
import random

"""
Sample input: "Aiden Chaoyang He is Co-founder of FedML, Inc., "
               "a Silicon Valley-based company building machine learning infrastructure to train, "
               "serve, and deploy AI models easily, economically, and securely, "
               "with holistic support of high-performance ML libraries, "
               "user-friendly AIOps, and a well-managed distributed GPU Cloud. "
Output: [{'entity_group': 'PER', 'score': 0.90398955, 'word': 'Aiden Chaoyang He', 'start': 0, 'end': 17}, 
        {'entity_group': 'ORG', 'score': 0.99320173, 'word': 'FedML, Inc', 'start': 35, 'end': 45}, 
        {'entity_group': 'MISC', 'score': 0.9811441, 'word': 'Silicon Valley', 'start': 50, 'end': 64}, 
        {'entity_group': 'MISC', 'score': 0.5548529, 'word': 'AI', 'start': 148, 'end': 150}, 
        {'entity_group': 'MISC', 'score': 0.62391496, 'word': 'AI', 'start': 264, 'end': 266}]
Note: https://huggingface.co/beki/en_spacy_pii_distilbert requires keys to call their API.
      PII dataset: https://huggingface.co/datasets/beki/privy
"""


def add_noise_to_string(original, noise_level=0.1):
    num_chars_to_change = int(len(original) * noise_level)
    possible_chars = string.ascii_letters + string.digits # + string.punctuation
    original_list = list(original)

    for _ in range(num_chars_to_change):
        position_to_change = random.randint(0, len(original) - 1)  # Select a random position
        new_char = random.choice(possible_chars)  # Select a random character to replace
        original_list[position_to_change] = new_char  # Replace the character at the selected position
    return ''.join(original_list)


class PIIProtector:
    def __init__(self):
        self.pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                 aggregation_strategy="simple")
        self.noise_type = "random"

    def add_mask(self, text):
        scores = self.pipeline(text)
        for d in scores:
            if d['score'] > 0.5:  # and d['entity_group'] in ['PER', 'ORG']:
                new_word = self.add_mask_for_string(d['word'])
                text = text[:d['start']] + new_word + text[d['end']:]
        return text

    def add_mask_for_string(self, token):
        if self.noise_type == "random":
            return add_noise_to_string(token, noise_level=0.5)
        else:
            raise ValueError("Not implemented!")


if __name__ == '__main__':
    protector = PIIProtector()
    print(protector.add_mask("Aiden Chaoyang He is Co-founder of FedML, Inc., "
                                   "a Silicon Valley-based company building machine learning infrastructure to train, "
                                   "serve, and deploy AI models easily, economically, and securely, "
                                   "with holistic support of high-performance ML libraries, "
                                   "user-friendly AIOps, and a well-managed distributed GPU Cloud."))
