from datasets import load_dataset
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from utils.constants import FOX_BASE_GPU, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
from inference.inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

TASK_NAME = ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
MODEL_NAME = FOX_BASE_GPU  # "google-bert/bert-base-uncased"

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    text = "i'm happy hahaha"
    dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")["train"]
    # dataset = prepare_toxic_chat_test_data()

    inference_engine = InferenceEngine(default_task=TASK_NAME, base_model=FOX_BASE_GPU,
                                       adapter_path="/fox_adapters/checkpoint-23000-all_in_one_unsafe_contents-final/")
    print(inference_engine.inference(text))
    # print(inference_engine.evaluation(texts=dataset["text"], labels=dataset["label"]))
