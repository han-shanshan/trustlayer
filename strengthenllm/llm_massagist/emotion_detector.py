from transformers import pipeline

"""
https://huggingface.co/michellejieli/emotion_text_classifier
emotions: anger, disgust, fear, joy, neutral, sadness, surprise
Example: "'I love this!'"
Detection score: {'label': 'joy', 'score': 0.9887555241584778}
"""


class EmotionDetector:
    def __init__(self):
        # self.pipeline = pipeline("sentiment-analysis") # distilbert-base-uncased-finetuned-sst-2-english
        self.pipeline = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

    def get_detection_results(self, text):
        scores = self.pipeline(text)
        print(f"detection scores: {scores}")
        if scores[0]['score'] > 0.5:
            print(f"the input {text} is detected as \"{scores[0]['label']}\"")


if __name__ == '__main__':
    detector = EmotionDetector()
    detector.get_detection_results("I love this!")
