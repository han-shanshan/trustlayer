from utils.constants import TOPIC_TASK_NAME, MODEL_NAME_BERT_BASE
from inference.inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

MODEL_NAME = MODEL_NAME_BERT_BASE  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TOPIC_TASK_NAME)
    # trainer.train()
    text = "i'm happy hahaha"
    inference_engine = InferenceEngine(task_name=TOPIC_TASK_NAME)
    print(inference_engine.inference(text))
