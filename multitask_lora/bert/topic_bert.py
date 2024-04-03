from multitask_lora.bert.constants import TOPIC_TASK_NAME
from multitask_lora.inference_engine import InferenceEngine
from multitask_lora.training_engine import TrainingEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

MODEL_NAME = "google-bert/bert-base-uncased"  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TOPIC_TASK_NAME)
    # trainer.train()

    text = "i'm happy hahaha"

    inference_engine = InferenceEngine(default_task=TOPIC_TASK_NAME)
    print(inference_engine.inference(text))
