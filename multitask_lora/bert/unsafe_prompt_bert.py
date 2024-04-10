from multitask_lora.constants import SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME
from multitask_lora.inference_engine import InferenceEngine
from multitask_lora.training_engine import TrainingEngine

TASK_NAME = UNSAFE_PROMPT_TASK_NAME
MODEL_NAME = "google-bert/bert-base-uncased"  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"
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
