from training.constants import SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, \
    MODEL_NAME_TINYLAMMA, TOXICITY_TASK_NAME, HALLUCINATION_TASK_NAME
from training.trust_inference_engine import TrustInferenceEngine
from training.training_engine import TrainingEngine
import os
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '6'

TASK_NAME = GIBBERISH_TASK_NAME
MODEL_NAME = MODEL_NAME_TINYLAMMA  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    trainer.train()
    text = "i'm happy hahaha"

    inference_engine = TrustInferenceEngine(default_task=TASK_NAME)
    print(inference_engine.inference(text))
