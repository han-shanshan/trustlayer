from hallucination_pipeline.hallucination_training_engine import HallucinationTrainingEngine
from multitask_lora.constants import MODEL_NAME_TINYLAMMA, CUSTOMIZED_HALLUCINATION_TASK_NAME
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

MODEL_NAME = MODEL_NAME_TINYLAMMA
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + CUSTOMIZED_HALLUCINATION_TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    trainer = HallucinationTrainingEngine(base_model_name=MODEL_NAME)
    trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
