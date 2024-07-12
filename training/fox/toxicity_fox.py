from utils.constants import TOXICITY_TASK, FOX_INSTRUCT
from training.training_engine import TrainingEngine
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

TASK_NAME = TOXICITY_TASK
MODEL_NAME = FOX_INSTRUCT  # "google-bert/bert-base-uncased"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections


Only use inputs as training dataset. When training toxicity classifier, select toxicity = 1 and jailbreaking = 0; 
when training unsafe prompts, select jailbreaking = 1 and toxicity = 0; 
https://huggingface.co/datasets/lmsys/toxic-chat 

"""

def train(config_path=""):
    config = json.load(open(config_path))
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    trainer.train()



if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
