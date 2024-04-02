import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, load_peft_weights, PeftConfig
import torch
import json
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

MODEL_NAME = "google-bert/bert-base-uncased"  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = "bert_peft_trainer"

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


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def train():
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded_dataset, id2label, label2id, label_dicts = get_encoded_datasets(tokenizer)

    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,  # "bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(label_dicts),
                                                               id2label=id2label,
                                                               label2id=label2id,
                                                               load_in_8bit=False
                                                               )

    print("=======start loading metric=========")
    # metric = evaluate.load("accuracy")
    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS  # this is necessary
        # inference_mode=True
    )
    # add LoRA adaptor
    # model.add_adapter(lora_config, adapter_name="semantic_adapter")
    model = get_peft_model(model, lora_config)
    print("=======print_trainable_parameters============")
    model.print_trainable_parameters()  # see % trainable parameters

    training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=1)
    batch_size = 8
    # training_args = TrainingArguments(
    #     OUTPUT_DIR,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     # metric_for_best_model=metric_name,
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
    # bert_peft_trainer.model.save_pretrained(lora_storage_path)
    model.save_pretrained(lora_storage_path)


def inf(text):
    # https://huggingface.co/docs/transformers/main/en/peft
    # https://github.com/huggingface/peft/discussions/661
    load_lora_path = OUTPUT_DIR + "/checkpoint-500"
    general_config_file_path = 'config.json'

    with open(general_config_file_path, 'r') as file:
        config = json.load(file)
    print(f"config = {config}")

    # lora_config = PeftConfig.from_pretrained(load_path)

    base_model = AutoModelForSequenceClassification.from_pretrained(config['base_model_name_or_path'],
                                                                    problem_type="multi_label_classification",
                                                                    num_labels=len(config['semantic']),
                                                                    id2label=config['semantic']
                                                                    )
    base_model.load_adapter(load_lora_path)

    # lora_weights = load_peft_weights(load_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # id2label = config['semantic']
    encoding = tokenizer(text, return_tensors="pt")

    # model = AutoModelForSequenceClassification.from_pretrained(load_path)
    encoding = {k: v.to(base_model.device) for k, v in encoding.items()}
    outputs = base_model(**encoding)
    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [config['semantic'][idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # train()
    text = "i'm happy hahaha"
    inf(text)
