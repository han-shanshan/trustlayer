from training.constants import GIBBERISH_TASK_NAME, FOX_BASE_GPU
from training.trust_inference_engine import TrustInferenceEngine
from training.training_engine import TrainingEngine
import os
import torch


# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

TASK_NAME = GIBBERISH_TASK_NAME
MODEL_NAME = FOX_BASE_GPU  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    # trainer.train(desired_total_data_n=100)
    trainer.train()
    # text = "i'm happy hahaha"

    # inference_engine = TrustInferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
