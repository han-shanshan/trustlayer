import numpy as np
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch



model_path = "multi_task_bert_single_dataset_test"

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
    encoded_dataset, id2label, label2id, label_dicts = get_encoded_datasets(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-mini",  # "bert-base-uncased",
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(label_dicts['semantics']),
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )
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


def get_encoded_datasets(tokenizer):
    tasks = ['semantics']
    datasets = {"semantics": load_dataset("sem_eval_2018_task_1", "subtask5.english")}
    # sentiment_dataset = concatenate_datasets([dataset["train"], dataset["test"], dataset["validation"]])
    label_dicts = {}
    # for task_name, dataset in datasets.items():
    #     print(task_name)
    #     print(datasets[task_name]["train"][0])
    label_dicts['semantics'] = [label for label in datasets['semantics']['train'].features.keys() if
                                label not in ['ID', 'Tweet']]
    id2label = {idx: label for idx, label in enumerate(label_dicts['semantics'])}
    label2id = {label: idx for idx, label in enumerate(label_dicts['semantics'])}

    def preprocess_data(examples):
        # take a batch of texts

        # print(f" len examles = ======== {len(examples)}, {type(examples)}")

        # print(examples[0])
        # print(examples[1])
        # print(examples[2])
        text = examples["Tweet"]

        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in label_dicts['semantics']}


        # print(f"labels = {label_dicts['semantics']}")
        print(f"examples.keys() = {examples.keys()}")
        # print(f"label batch = {labels_batch}")


        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(label_dicts['semantics'])))
        # fill numpy array
        for idx, label in enumerate(label_dicts['semantics']):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding
    
    

    encoded_dataset = datasets['semantics'].map(preprocess_data, batched=True,
                                                remove_columns=datasets['semantics']['train'].column_names)
    # print(f"example = {encoded_dataset['train'][0]}")

    # print(f"print dataset ----- {datasets['semantics'][0]}")
    # print(f"print features_dict ----- {features_dict[task_name][phase][0]}")
    print(f"len =========={len(encoded_dataset)}")

    # print(encoded_dataset['train'][0].keys())  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # print(f"labels = {encoded_dataset['train'][0]['labels']}")
    # print(tokenizer.decode(encoded_dataset['train'][0]['input_ids']))
    # print
    # 
    # d2label[idx] for idx, label in enumerate(encoded_dataset['train'][0]['labels']) if label == 1.0]}")
    encoded_dataset.set_format("torch")
    return encoded_dataset, id2label, label2id, label_dicts


def inf(text):
    load_path = model_path + "/checkpoint-4280"
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
    
    train_multi_task_bert()
    

    text = "i'm happy hahaha"
    inf(text)












