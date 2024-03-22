import numpy as np
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import AutoModelForSequenceClassification
from bert_model import BertForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

MODEL_PATH = "multi_task_bert_test"
TASKS = ['semantics']
MODEL_NAME = "prajjwal1/bert-mini"  # "bert-base-uncased"

# 16 * 7B *
# 1 模型 1 gradient optimization 2
# + 16 * 7 +
#
# 4*16*

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


def load_a_dataset(task_type):
    if task_type == "semantics":
        dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        return dataset, [label for label in dataset['train'].features.keys() if
                         label not in ['ID', 'Tweet']]


def preprocess_semantic_data(examples, labels):
    # take a batch of texts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = examples["Tweet"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


convert_func_dict = {
    "semantics": preprocess_semantic_data,
}

columns_dict = {
        "semantics": ["input_ids", "attention_mask", "labels"],
        # "spaadia_squad_pairs": ["input_ids", "attention_mask", "labels"],
}


def get_encoded_datasets():
    datasets = {}
    # sentiment_dataset = concatenate_datasets([dataset["train"], dataset["test"], dataset["validation"]])
    label_dicts = {}
    for task in TASKS:
        datasets[task], label_dicts[task] = load_a_dataset(task)

    print(f"datasets['semantics']['train'][0] = {datasets['semantics']['train'][0]}")

    features_dict = {}
    id2label_dict = {}
    for task_name in TASKS:
        # features_dict[task_name] = {
        features_dict[task_name] = datasets[task_name].map(
            lambda X: convert_func_dict[task](X, label_dicts[task_name]),
            batched=True, remove_columns=datasets[task_name]['train'].column_names)
        features_dict[task_name].set_format(type="torch")
        id2label_dict[task_name] = {idx: label for idx, label in enumerate(label_dicts[task_name])}
        # print(f"id2label_dict[task_name] = {id2label_dict[task_name]}")

        # print(f"print dataset ----- {datasets[task_name][phase][0]}")
        # print(f"print features_dict ----- {features_dict[task_name][phase][0]}")





    # id2label = {idx: label for idx, label in enumerate(label_dicts['semantics'])}
    # label2id = {label: idx for idx, label in enumerate(label_dicts['semantics'])}

    # encoded_dataset = datasets['semantics'].map(preprocess_data, batched=True,
    # #                                             remove_columns=datasets['semantics']['train'].column_names)
    # encoded_dataset = datasets['semantics'].map(lambda examples: preprocess_data(examples, extra_param),
    #                                             batched=True,
    #                                             remove_columns=datasets['semantics']['train'].column_names)

    # print(f"example = {encoded_dataset['train'][0]}")
    # # print(encoded_dataset['train'][0].keys())  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # print(f"labels = {encoded_dataset['train'][0]['labels']}")
    # print(tokenizer.decode(encoded_dataset['train'][0]['input_ids']))
    # encoded_dataset.set_format("torch")
    return features_dict, id2label_dict


def train_multi_task_bert():
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    features_dict, id2label_dict = get_encoded_datasets()
    task_labels_map = {}
    for task in TASKS:
        task_labels_map[task] = len(id2label_dict[task])

    # train_dataset = {
    #     task_name: features_dict[task_name]["train"] for task_name in features_dict.keys()
    # }
    train_datasets = [features_dict[task_name]["train"] for task_name in features_dict.keys()]
    unified_train_data = concatenate_datasets(train_datasets)
    
    # validation_dataset = {
    #     task_name: dataset[task_name]["validation"] for task_name, dataset in features_dict.items()
    # }

    # print(f"========== encoded_dataset[train] = {encoded_dataset['train']}")
    print(f"========== train_dataset = {unified_train_data}")


    # encoded_dataset, id2label, label2id, label_dicts = get_encoded_datasets2(tokenizer)



    model = BertForSequenceClassification.from_pretrained(MODEL_NAME,  # "bert-base-uncased",
                    task_labels_map=task_labels_map,
                    # num_labels=len(id2label_dict['semantics']),
                    # problem_type="multi_label_classification",
                    )

    batch_size = 8
    metric_name = "f1"
    args = TrainingArguments(
        MODEL_PATH,
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
    trainer = Trainer(
        model,
        args,
        train_dataset=unified_train_data,
        # eval_dataset=validation_dataset,
        # tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.train()
    # trainer.evaluate()
    model.save_pretrained(MODEL_PATH)
    return model


def inf(text):
    load_path = MODEL_PATH + "/checkpoint-22256"
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    config = AutoConfig.from_pretrained(load_path)
    id2label = config.id2label
    encoding = tokenizer(text, return_tensors="pt")

    model = BertForSequenceClassification.from_pretrained(load_path)
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

    # text = "i'm happy hahaha"
    # inf(text)
