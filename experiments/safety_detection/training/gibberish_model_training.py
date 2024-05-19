from training.constants import GIBBERISH_TASK_NAME, FOX_BASE_GPU
from training.trust_inference_engine import TrustInferenceEngine
from training.training_engine import TrainingEngine
import wandb
import os
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = GIBBERISH_TASK_NAME
MODEL_NAME = FOX_BASE_GPU  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    wandb.init(project=f"{GIBBERISH_TASK_NAME} with FOX")
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME, config={"metrics_average": "macro"})
    trainer.train()
    # text = "i'm happy hahaha"

    # inference_engine = TrustInferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
