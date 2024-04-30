from multitask_lora.constants import TOXICITY_TASK_NAME, FOX_BASE_GPU
from multitask_lora.trust_inference_engine import TrustInferenceEngine
from multitask_lora.training_engine import TrainingEngine
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

TASK_NAME = TOXICITY_TASK_NAME
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
