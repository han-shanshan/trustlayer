from safety_detection.securellm.detector_base import BaseTrustDetector
from safety_detection.securellm.toxicity.toxicity_detector_toxic_bert import ToxicityDetectorToxicBert

from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

"""
dataset: https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi
twitter topic classification: train Bert using twitter topic dataset && politics dataset

model: https://huggingface.co/cardiffnlp/tweet-topic-21-multi 
"""


class PoliticalRiskDetector(BaseTrustDetector):
    def __init__(self):
        # self.pipeline = pipeline("sentiment-analysis") # distilbert-base-uncased-finetuned-sst-2-english
        self.toxicity_detector = ToxicityDetectorToxicBert()
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi")
        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def get_detection_results(self, input_or_output_text, text2=None):
        if not self.toxicity_detector.get_is_toxic(input_or_output_text):
            print(f"the input \"{input_or_output_text}\" does not have political risks")
            return

        score = self.pipeline(input_or_output_text)[0]
        print(f"detection scores: {score}")
        if score['score'] > 0.5 and score['label'] == 'news_&_social_concern':
            print(f"the input \"{input_or_output_text}\" has political risks")
        else:
            print(f"the input \"{input_or_output_text}\" does not have political risks")


if __name__ == '__main__':
    detector = PoliticalRiskDetector()
    detector.get_detection_results("He is one of the stupidest presidents in US history")
