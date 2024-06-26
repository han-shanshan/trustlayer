from utils.constants import FOX, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
from inference.inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
MODEL_NAME = FOX  # "google-bert/bert-base-uncased"

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    text = "i'm happy hahaha"

    inference_engine = InferenceEngine(task_name=TASK_NAME, base_model=FOX,
                                       adapter_path="/fox_adapters/checkpoint-23000-all_in_one_unsafe_contents-final/")
    print(inference_engine.inference([text]))
