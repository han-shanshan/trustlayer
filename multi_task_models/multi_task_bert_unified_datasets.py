import numpy as np
from datasets import load_dataset, DatasetDict
from datasets import concatenate_datasets
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from constants import SEMANTIC_TASK_NAME, TOPIC_TASK_NAME

from data_processor import DataProcessor



model_path = "multi_task_bert_model_unified_datasets"

"""
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

    DatasetDict({
    train: Dataset({
        features: ['ID', 'Tweet', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'],
        num_rows: 6838
    })
    test: Dataset({
        features: ['ID', 'Tweet', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'],
        num_rows: 3259
    })
    validation: Dataset({
        features: ['ID', 'Tweet', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'],
        num_rows: 886
    })
    })
"""


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


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def train_multi_task_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_dataset, id2label, label2id, labels = get_encoded_datasets(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-mini",  # "bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )
    
    print(f"labels = {labels},id2label = {id2label}, label2id = {label2id}")
    
    batch_size = 8
    metric_name = "f1"
    args = TrainingArguments(
        model_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
    )

    # print(f"========== encoded_dataset[train] = {encoded_dataset['train']}")

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    # model.save_pretrained(model_path)
    return model

def get_phase_names(task_name):
    if task_name is TOPIC_TASK_NAME:
        return "train_2020", "test_2020", "validation_2020"
    else:
        return "train", "test", "validation"



def get_remove_column_names(datasets, task_name):
    train_phase_name, _, _ = get_phase_names(task_name=task_name)
    return datasets[task_name][train_phase_name].column_names

def get_datasets(tasks):
    datasets = {}
    for task_name in tasks:
        if task_name is TOPIC_TASK_NAME:
            datasets[task_name] = load_dataset("cardiffnlp/tweet_topic_multi")
        elif task_name is SEMANTIC_TASK_NAME:
            datasets[task_name] = load_dataset("sem_eval_2018_task_1", "subtask5.english")
    return datasets


def get_encoded_datasets(tokenizer):
    data_processor = DataProcessor(tokenizer=tokenizer)
    tasks = [TOPIC_TASK_NAME, SEMANTIC_TASK_NAME]
    datasets = get_datasets(tasks=tasks)

    label_name_dicts = {}
    id2labels = {}
    label2ids = {}
    for task_name, dataset in datasets.items():
        label_name_dicts[task_name] = data_processor.prepare_label_dict_for_a_task(task_name, dataset) 
    idx = 0
    labels = []
    for task_name in tasks:
        for label in label_name_dicts[task_name]:
            if label not in labels:
                labels.append(label)
                id2labels[idx] = label
                label2ids[label] = idx 
                idx += 1

    data_processor.set_labels(labels=labels)

    encoded_datasets = {}
    D = {}
    for task_name in tasks:
        data_processor.set_task_name(task_name)
        encoded_datasets[task_name] = datasets[task_name].map(data_processor.process_data_unified_datasets,    # preprocess_data, 
                        batched=True, remove_columns=get_remove_column_names(datasets, task_name))
    
        for phase in encoded_datasets[task_name].keys():
            # Check if the phase already exists in D
            original_phase_name = phase
            if phase not in ["train", "test", "validation"]:  # in case in some datasets the phases are called different names, e.g., "train_2020"
                for p in ["train", "test", "validation"]:
                    if p in phase:
                        phase = p
                        break
                
            if phase in D:
                # Concatenate the new dataset with the existing one
                D[phase] = concatenate_datasets([D[phase], encoded_datasets[task_name][original_phase_name]])
            else:
                D[phase] = encoded_datasets[task_name][original_phase_name]


    final_dataset = DatasetDict(D)
    # d2label[idx] for idx, label in enumerate(encoded_dataset['train'][0]['labels']) if label == 1.0]}")
    final_dataset.set_format("torch")
    return final_dataset, id2labels, label2ids, data_processor.get_labels()


def inf(text):
    load_path = model_path + "/checkpoint-21910"
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    config = AutoConfig.from_pretrained(load_path)
    id2label = config.id2label
    encoding = tokenizer(text, return_tensors="pt")

    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    # print(f"logits.shape = {logits.shape}")

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)


if __name__ == '__main__':
    

    # train_multi_task_bert()
    

    # text = "i'm angry because i hate the US president. He is stupid"
    text = "The athletes are promoting awareness for climate change"
    inf(text)

