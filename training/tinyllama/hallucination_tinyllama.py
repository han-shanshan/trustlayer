from utils.constants import MODEL_NAME_TINYLAMMA, HALLUCINATION_TASK
from inference.classification_inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = HALLUCINATION_TASK
MODEL_NAME = MODEL_NAME_TINYLAMMA  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    # trainer.train()
    text = "i'm happy hahaha"

    inference_engine = InferenceEngine(task_name=TASK_NAME)
    print(inference_engine.inference([text, text]))
