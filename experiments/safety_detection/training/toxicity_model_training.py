from utils.constants import TOXICITY_TASK, FOX_INSTRUCT
from training.training_engine import TrainingEngine
import os
import wandb
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

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# # Example usage
# text = "He got his money... now he lies in wait till after the election in 2 yrs.... dirty politicians need to be afraid of Tar and feathers again... but they aren't and so the people get screwed."
# scores = predict_scores(text)
# print("Scores:", scores)

if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    wandb.init(project=f"{TASK_NAME} with FOX")
    trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME,
                             config={"metrics_average": "macro", "dataset_type": ["sophisticated"]})
    trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
