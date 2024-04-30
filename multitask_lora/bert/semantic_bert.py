from multitask_lora.constants import SEMANTIC_TASK_NAME, MODEL_NAME_BERT_BASE
from multitask_lora.trust_inference_engine import TrustInferenceEngine
from multitask_lora.training_engine import TrainingEngine

TASK_NAME = "semantic"
MODEL_NAME = MODEL_NAME_BERT_BASE  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + TASK_NAME


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # trainer = TrainingEngine(base_model_name=MODEL_NAME, task_name=SEMANTIC_TASK_NAME)
    # trainer.train()
    text = "i'm happy hahaha"

    inference_engine = TrustInferenceEngine(default_task=TASK_NAME)
    print(inference_engine.inference(text))
