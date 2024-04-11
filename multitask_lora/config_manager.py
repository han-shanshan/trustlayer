from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments

from multitask_lora.constants import MODEL_NAME_TINYLAMMA


class ConfigManager:
    def __init__(self, task_name, model):
        self.task_name = task_name
        self.model = model

    def get_training_config(self, output_dir, batch_size=8):
        if self.model is MODEL_NAME_TINYLAMMA:
            return TrainingArguments(
                output_dir=output_dir,  # directory to save and repository id
                num_train_epochs=3,  # number of training epochs
                per_device_train_batch_size=3,  # batch size per device during training
                gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
                gradient_checkpointing=True,  # use gradient checkpointing to save memory
                optim="adamw_torch_fused",  # use fused adamw optimizer
                # logging_steps=10,  # log every 10 steps
                save_strategy="epoch",  # save checkpoint every epoch
                learning_rate=2e-4,  # learning rate, based on QLoRA paper
                bf16=True,  # use bfloat16 precision
                tf32=True,  # use tf32 precision
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
                lr_scheduler_type="constant",  # use constant learning rate scheduler
            )

        return TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=50,
                weight_decay=0.01,
                load_best_model_at_end=True,
                # metric_for_best_model=metric_name,
                # push_to_hub=True,
            )

    def get_lora_config(self):
        if self.model is MODEL_NAME_TINYLAMMA:
            return LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
