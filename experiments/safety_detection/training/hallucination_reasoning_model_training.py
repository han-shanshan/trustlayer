import os
from training.hallucination_reasoning_training_engine import HallucinationReasoningTrainingEngine
import torch
import json
from utils.constants import HALLUCINATION_REASONING_TASK, FOX_INSTRUCT
import wandb
from datasets import load_dataset

TASK_NAME = HALLUCINATION_REASONING_TASK
MODEL_NAME = FOX_INSTRUCT

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections


Only use inputs as training dataset. When training toxicity classifier, select toxicity = 1 and jailbreaking = 0; 
when training unsafe prompts, select jailbreaking = 1 and toxicity = 0; 
https://huggingface.co/datasets/lmsys/toxic-chat 

# """


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    run_id = wandb.init(project=f"{TASK_NAME} with FOX")
    torch.cuda.empty_cache()
    dataset_types = [
        # "rag-hallucination1000", # 1000 in total
        "HaluEval-qa",
        "HaluEval-dialogue",
        "HaluEval-summarization"
    ]
    data_num_dict = {
        "HaluEval-qa": {"train": 8000, "validation": 1000, "test": 1000},
        "HaluEval-dialogue": {"train": 8000, "validation": 1000, "test": 1000},
        "HaluEval-summarization": {"train": 8000, "validation": 1000, "test": 1000},
        # "rag-hallucination1000": {"train": 500, "validation": 20, "test": 0},
    }

    trainer = HallucinationReasoningTrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME,
                                                   config={"dataset_types": dataset_types,
                                                           "data_num_dict": data_num_dict})
    trainer.process(batch_size=1)
    # trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
