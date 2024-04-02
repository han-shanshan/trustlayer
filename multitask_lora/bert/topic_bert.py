import json

import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, load_peft_weights
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datasets import concatenate_datasets
from multitask_lora.bert.constants import TOPIC_TASK_NAME
from multitask_lora.bert.data_processor import DataProcessor

TASK_NAME = TOPIC_TASK_NAME
MODEL_NAME = "google-bert/bert-base-uncased"  # "prajjwal1/bert-small" #"nlptown/bert-base-multilingual-uncased-sentiment" # "prajjwal1/bert-small"
lora_storage_path = MODEL_NAME.split("/")[1] + "_" + TASK_NAME
OUTPUT_DIR = "bert_peft_trainer_" + TASK_NAME


def get_encoded_datasets(tokenizer):
    data_processor = DataProcessor(tokenizer=tokenizer)
    data_processor.task_name = TASK_NAME
    dataset = load_dataset("cardiffnlp/tweet_topic_multi")

    label_names = data_processor.prepare_label_dict_for_a_task( dataset)
    print(f"label names = {label_names}")

    idx = 0
    id2labels = {}
    label2ids = {}
    for label in label_names:
        id2labels[idx] = label
        label2ids[label] = idx
        idx += 1

    # data_processor.set_labels(labels=labels)
    data_processor.set_labels(labels=label_names)

    encoded_datasets = {}
    D = {}
    data_processor.labels = label_names

    data_processor.set_task_name(TASK_NAME)
    encoded_dataset = dataset.map(data_processor.process_data_diff_datasets,  # preprocess_data,
                    batched=True, remove_columns=data_processor.get_remove_column_names(dataset))

    for phase in encoded_dataset.keys():
        # Check if the phase already exists in D
        original_phase_name = phase
        if phase not in ["train", "test",
                         "validation"]:  # in case in some datasets the phases are called different names, e.g., "train_2020"
            for p in ["train", "test", "validation"]:
                if p in phase:
                    phase = p
                    break
        if phase in D:
            # Concatenate the new dataset with the existing one
            D[phase] = concatenate_datasets([D[phase], encoded_dataset[original_phase_name]])
        else:
            D[phase] = encoded_dataset[original_phase_name]

    final_dataset = DatasetDict(D)
    # d2label[idx] for idx, label in enumerate(encoded_dataset['train'][0]['labels']) if label == 1.0]}")
    final_dataset.set_format("torch")
    return final_dataset, id2labels, label2ids, label_names


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

#
# def inf(text):
#     load_path = OUTPUT_DIR + "/checkpoint-500"
#
#     # config = AutoConfig.from_pretrained(load_path)
#
#     lora_weights = load_peft_weights(load_path)
#
#     tokenizer = AutoTokenizer.from_pretrained(load_path)
#     id2label = config.id2label
#     encoding = tokenizer(text, return_tensors="pt")
#
#     model = AutoModelForSequenceClassification.from_pretrained(load_path)
#     encoding = {k: v.to(model.device) for k, v in encoding.items()}
#     outputs = model(**encoding)
#     logits = outputs.logits
#
#     # apply sigmoid + threshold
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(logits.squeeze().cpu())
#     predictions = np.zeros(probs.shape)
#     predictions[np.where(probs >= 0.5)] = 1
#     # turn predicted id's into actual label names
#     predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
#     print(predicted_labels)


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
                                                                    num_labels=len(config['topic']),
                                                                    id2label=config['topic']
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
    predicted_labels = [config['topic'][idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/main/en/peft
    # train()
    text = "i'm happy hahaha"
    inf(text)
