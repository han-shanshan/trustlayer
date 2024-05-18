from training.constants import UNSAFE_PROMPT_TASK_NAME, FOX_BASE_GPU
from training.trust_inference_engine import TrustInferenceEngine
from training.training_engine import TrainingEngine
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TASK_NAME = UNSAFE_PROMPT_TASK_NAME
MODEL_NAME = FOX_BASE_GPU  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
