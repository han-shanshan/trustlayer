import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from multitask_lora.constants import MODEL_NAME_BERT_BASE


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    model_name = MODEL_NAME_BERT_BASE  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    dataset = load_dataset("yelp_review_full")
    small_train_dataset = dataset["train"].shuffle(seed=4).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=4).select(range(1000))

    print("=======start mapping datasets=========")
    tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)
    print(f"tokenized_train_dataset: {tokenized_train_dataset[0]}")

    # small_train_dataset = small_train_dataset.remove_columns(dataset["train"].column_names)
    # small_eval_dataset = small_eval_dataset.remove_columns(dataset["train"].column_names)

    print("=======start loading metric=========")

    # metric = evaluate.load("accuracy")

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # this is necessary
        inference_mode=True
    )

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    print("=======print_trainable_parameters============")
    model.print_trainable_parameters()  # see % trainable parameters

    training_args = TrainingArguments(output_dir="bert_peft_trainer", num_train_epochs=1)
    bert_peft_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,  # training dataset requires column input_ids
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )
    bert_peft_trainer.train()
    lora_storage_path = model_name.split("/")[1]
    bert_peft_trainer.model.save_pretrained(lora_storage_path)
