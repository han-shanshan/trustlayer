from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from utils.constants import HALLUCINATION_EXPLANATION_TASK_NAME, MODEL_NAME_TINYLAMMA, FOX


class TrainingConfigManager:
    def __init__(self, task_name=None, model=None, config=None):
        self.task_name = task_name
        self.model = model
        self.config = config

    @staticmethod
    def get_training_config(output_dir, task_name, batch_size=8):
        if task_name == HALLUCINATION_EXPLANATION_TASK_NAME:
            return TrainingArguments(
                output_dir=output_dir,  # directory to save and repository id
                num_train_epochs=10,  # number of training epochs
                per_device_train_batch_size=batch_size,  # batch size per device during training
                # max_steps=1,  # for testing
                gradient_accumulation_steps=10,  # number of steps before performing a backward/update pass
                gradient_checkpointing=False,  # use gradient checkpointing to save memory
                optim="adamw_torch_fused",  # use fused adamw optimizer
                logging_steps=10,  # log every 10 steps
                save_strategy="epoch",  # save checkpoint every epoch
                evaluation_strategy="epoch",
                learning_rate=5e-5,  # learning rate, based on QLoRA paper
                bf16=True,  # use bfloat16 precision
                # tf32=True,  # use tf32 precision
                # fp16=True,
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
                lr_scheduler_type="linear",
                report_to=["wandb"],
                load_best_model_at_end=True
            )
        else:
            return TrainingArguments(
                output_dir=output_dir,  # directory to save and repository id
                num_train_epochs=10,  # number of training epochs
                per_device_train_batch_size=batch_size,  # batch size per device during training
                gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
                gradient_checkpointing=True,  # use gradient checkpointing to save memory
                optim="adamw_torch_fused",  # use fused adamw optimizer
                logging_steps=10,  # log every 10 steps
                save_strategy="epoch",  # save checkpoint every epoch
                evaluation_strategy="epoch",
                learning_rate=5e-4,  # learning rate, based on QLoRA paper
                bf16=True,  # use bfloat16 precision
                tf32=True,  # use tf32 precision
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
                lr_scheduler_type="linear",
                report_to=["wandb"],
                load_best_model_at_end=True
            )

        # return TrainingArguments(
        #     output_dir=output_dir,
        #     evaluation_strategy="epoch",
        #     save_strategy="epoch",
        #     learning_rate=2e-5,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     num_train_epochs=20,
        #     weight_decay=0.01,
        #     load_best_model_at_end=True,
        #     # metric_for_best_model=metric_name,
        #     # push_to_hub=True,
        # )

    @staticmethod
    def get_lora_config(model_name):
        if model_name in [MODEL_NAME_TINYLAMMA, FOX]:
            return LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
        # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
