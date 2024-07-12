from utils.constants import GIBBERISH_TASK, FOX_INSTRUCT
from inference.inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

TASK_NAME = GIBBERISH_TASK
MODEL_NAME = FOX_INSTRUCT
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    # trainer.train()
    text = "i'm happy hahaha"

    inference_engine = InferenceEngine(task_name=TASK_NAME)
    print(inference_engine.inference(text))
