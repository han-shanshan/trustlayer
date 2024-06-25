# from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from utils.constants import FOX, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
from inference.inference_engine import InferenceEngine
import os
from datasets import load_dataset


os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

TASK_NAME = ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
MODEL_NAME = FOX

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    # text = "hello"
    # text = "i'm happpy hahaha"
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")["train"]
    # dataset = load_dataset('csv', data_files="./fox_adapters/dulcet-bee-14/all_in_one_unsafe_contents_test_data.csv")["train"]
    # print(f"dataset size = {len(dataset)}")

    dataset = prepare_toxic_chat_test_data()


    inference_engine = InferenceEngine(default_task=TASK_NAME, base_model=FOX,  #adapter_path=None)
                                       adapter_path="./fox_adapters/checkpoint-23000-all_in_one_unsafe_contents-2024-06-05 12:50:39.857450-final")
                                    #    adapter_path="./fox_adapters/raid-toxicity-final")
                                    #    adapter_path="./fox_adapters/dulcet-bee-14/checkpoint-23000-all_in_one_unsafe_contents-final/")
    # print(inference_engine.inference(text))
    # counter = 0
    # print(dataset["text"][:5])
    # print(dataset["label"][:5])
    # print(dataset)
    # for text in dataset["text"][:5]:
    #     print(f", text = {text}")
    #     print(inference_engine.inference(text))
    #     counter += 1
    # print(inference_engine.evaluation(texts=dataset["text"], labels=dataset["label"]))
    print(inference_engine.evaluate(dataset=dataset))
