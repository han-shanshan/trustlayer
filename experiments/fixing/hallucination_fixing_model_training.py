from training.hallucination_fixing_training_engine import HallucinationFixingTrainingEngine
from training.hallucination_reasoning_training_engine import HallucinationReasoningTrainingEngine
import torch
import json
from utils.constants import HALLUCINATION_FIXING_TASK, FOX_INSTRUCT
import wandb
from datasets import load_dataset
import os
from transformers import EarlyStoppingCallback, AutoModelForCausalLM
from peft import get_peft_model
from training.training_config_manager import TrainingConfigManager
from utils.util import get_tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


TASK_NAME = HALLUCINATION_FIXING_TASK
MODEL_NAME = FOX_INSTRUCT


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
#
#
# def cleanup():
#     torch.distributed.destroy_process_group()
#
#
# def get_pretrained_model(base_model_name):
#     model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_8bit=False,
#                                                  # device_map="auto",
#                                                  torch_dtype=torch.float32,
#                                                  trust_remote_code=True)
#     model.config.pad_token_id = model.config.eos_token_id
#     model.enable_input_require_grads()
#     # this line solves the bug: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#     # model.config.use_cache = False
#
#     model = get_peft_model(model, TrainingConfigManager.get_lora_config(model_name=self.base_model_name))
#     model.print_trainable_parameters()  # see % trainable parameters
#
#     return model
#
# def main(rank, world_size):
#     setup(rank, world_size)
#
#     # Load model and tokenizer
#     # model = get_pretrained_model(MODEL_NAME)
#     # tokenizer = get_tokenizer(MODEL_NAME)
#
#     # Move model to GPU
#     # model.to(rank)
#     #
#     # # Wrap model in DDP
#     # model = DDP(model, device_ids=[rank])
#
#     # Load and prepare your dataset here
#     dataset =
#
#     # Create DistributedSampler
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
#
#     # Create DataLoader
#     train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
#
#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         logging_dir="./logs",
#     )
#
#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
#                                     'attention_mask': torch.stack([f['attention_mask'] for f in data]),
#                                     'labels': torch.stack([f['label'] for f in data])},
#     )
#
#     # Start training
#     trainer.train()
#
#     cleanup()


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    run_id = wandb.init(project=f"{TASK_NAME} with FOX-new")
    torch.cuda.empty_cache()
    dataset_types = [
        # "rag-hallucination1000", # 1000 in total
        "HaluEval-qa",
        "HaluEval-dialogue",
        # "HaluEval-summarization"
    ]
    data_num_dict = {
        "HaluEval-qa": {"train": 8000, "validation": 1000, "test": 1000},
        "HaluEval-dialogue": {"train": 8000, "validation": 1000, "test": 1000},
        # "HaluEval-summarization": {"train": 8000, "validation": 1000, "test": 1000},
        # "rag-hallucination1000": {"train": 500, "validation": 20, "test": 0},
    }

    trainer = HallucinationFixingTrainingEngine(base_model_name=MODEL_NAME, task_name=TASK_NAME,
                                                config={"dataset_types": dataset_types,
                                                        "data_num_dict": data_num_dict})
    trainer.process(batch_size=8)
    # trainer.train()
    # text = "i'm happy hahaha"
    #
    # inference_engine = InferenceEngine(default_task=TASK_NAME)
    # print(inference_engine.inference(text))
