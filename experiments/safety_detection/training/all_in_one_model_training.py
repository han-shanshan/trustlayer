
from training.training_engine import TrainingEngine
import os
from utils.constants import ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME, FOX
import wandb
from datasets import load_dataset
import torch


TASK_NAME = ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
MODEL_NAME = FOX  # "google-bert/bert-base-uncased"

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections


Only use inputs as training dataset. When training toxicity classifier, select toxicity = 1 and jailbreaking = 0; 
when training unsafe prompts, select jailbreaking = 1 and toxicity = 0; 
https://huggingface.co/datasets/lmsys/toxic-chat 

"""

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    torch.cuda.empty_cache()
    run_id = wandb.init(project=f"{TASK_NAME} with FOX-New")
 
    dataset_types = [
        "mt-bench", "HEx-PHI", # "toxic-chat",
        "openai", "hotpot_qa", 
        "truthful_qa", 
        "awesome_chatgpt_prompts", "jigsaw",
        #  "gibberish", 
        "gpt-jailbreak", 
        "jailbreak",
        "personalization_prompt", "qa-chat-prompts",
        "chatgpt-prompts", "10k_prompts_ranked",
        "iterative-prompt",
        "instruction-following"
        ]

    data_num_dict = {
        "HEx-PHI": {"train": 330, "validation": 0, "test": 0},
        # "toxic-chat": {"train": 0, "validation": 200, "test": 0},
        "openai": {"train": 160, "validation": 1500, "test": 0},
        "hotpot_qa": {"train": 3000, "validation": 2500, "test": 500},
        "truthful_qa": {"train": 500, "validation": 100, "test": 100},
        "awesome_chatgpt_prompts": {"train": 0, "validation": 150, "test": 0},
        "jigsaw": {"train": 50000, "validation": 2000, "test": 300},
        # "gibberish": {"train": 1000, "validation": 150, "test": 100},
        "mt-bench": {"train": 0, "validation": 80, "test": 0},
        "gpt-jailbreak": {"train": 0, "validation": 78, "test": 0},
        "jailbreak": {"train": 400, "validation": 0, "test": 70},
        "personalization_prompt": {"train": 1000, "validation": 800, "test": 200},
        "qa-chat-prompts": {"train": 0, "validation": 200, "test": 0},
        "chatgpt-prompts": {"train": 350, "validation": 0, "test": 0},
        "10k_prompts_ranked": {"train": 500, "validation": 500, "test": 200},
        "iterative-prompt": {"train": 500, "validation": 500, "test": 200},
        "instruction-following": {"train": 200, "validation": 340, "test": 0},
        }

    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME,
                             config={"metrics_average": "macro", "dataset_types": dataset_types,
                                     "data_num_dict": data_num_dict})
    trainer.train(batch_size=32)
    # trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
