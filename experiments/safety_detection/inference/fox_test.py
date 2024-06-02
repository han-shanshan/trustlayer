from training.constants import FOX_BASE_GPU, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
from training.trust_inference_engine import TrustInferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
MODEL_NAME = FOX_BASE_GPU  # "google-bert/bert-base-uncased"

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME)
    # trainer.train()
    text = "i'm happy hahaha"

    inference_engine = TrustInferenceEngine(default_task=TASK_NAME)
    print(inference_engine.inference([text]))
