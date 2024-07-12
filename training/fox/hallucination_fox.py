from utils.constants import HALLUCINATION_TASK, FOX_INSTRUCT
from inference.inference_engine import InferenceEngine
from training.training_engine import TrainingEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = HALLUCINATION_TASK
MODEL_NAME = FOX_INSTRUCT  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    trainer.process()
    text = "i'm happy hahaha"

    inference_engine = InferenceEngine(task_name=TASK_NAME)
    print(inference_engine.inference([text, text]))
