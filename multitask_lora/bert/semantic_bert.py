import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


def get_encoded_datasets(tokenizer):
    datasets = load_dataset("sem_eval_2018_task_1", "subtask5.english")
    # sentiment_dataset = concatenate_datasets([dataset["train"], dataset["test"], dataset["validation"]])
    label_dicts = {}
    # for task_name, dataset in datasets.items():
    #     print(task_name)
    #     print(datasets[task_name]["train"][0])
    label_dicts = [label for label in datasets['train'].features.keys() if
                                label not in ['ID', 'Tweet']]
    id2label = {idx: label for idx, label in enumerate(label_dicts)}
    label2id = {label: idx for idx, label in enumerate(label_dicts)}

    def preprocess_data(examples):
        # take a batch of texts
        text = examples["Tweet"]

        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in label_dicts}

        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(label_dicts)))
        # fill numpy array
        for idx, label in enumerate(label_dicts):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    encoded_dataset = datasets.map(preprocess_data, batched=True,
                                                remove_columns=datasets['train'].column_names)
    encoded_dataset.set_format("torch")
    return encoded_dataset, id2label, label2id, label_dicts


def train():
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    model_name = "google-bert/bert-base-uncased"  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded_dataset, id2label, label2id, label_dicts = get_encoded_datasets(tokenizer)

    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    model = AutoModelForSequenceClassification.from_pretrained(model_name,  # "bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(label_dicts),
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )

    print("=======start loading metric=========")
    metric = evaluate.load("accuracy")
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

    # args = TrainingArguments(
    #     model_path,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=10,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=metric_name,
    #     # push_to_hub=True,
    # )

    bert_peft_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    bert_peft_trainer.train()
    lora_storage_path = model_name.split("/")[1]
    bert_peft_trainer.model.save_pretrained(lora_storage_path)


if __name__ == '__main__':
    train()
