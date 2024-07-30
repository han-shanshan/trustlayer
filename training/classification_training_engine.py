from data_operation.data_loader import DataLoader
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainerCallback, \
    TrainerState, TrainerControl
from transformers import Trainer
from peft import get_peft_model
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine
from utils.constants import GIBBERISH_TASK, UNSAFE_PROMPT_TASK, HALLUCINATION_TASK, \
    TOXICITY_TASK, MODEL_NAME_TINYLAMMA, FOX_INSTRUCT, SEMANTIC_TASK, TOPIC_TASK, \
    ALL_IN_ONE_UNSAFE_CONTENTS_TASK
from data_operation.data_processor import DataProcessor
import evaluate
from utils.file_operations import write_hf_dataset_to_csv
from scipy.special import expit as sigmoid
from datetime import datetime
from datasets import DatasetDict

# accuracy = evaluate.load("accuracy")
from utils.util import get_tokenizer

accuracy_metric = load_metric("accuracy", trust_remote_code=True)
precision_metric = load_metric("precision", trust_remote_code=True)
recall_metric = load_metric("recall", trust_remote_code=True)
f1_metric = load_metric("f1", trust_remote_code=True)
roc_auc_metric = evaluate.load("roc_auc", trust_remote_code=True)


def compute_metrics(labels, predictions, probabilities, metrics_average="macro"):
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    # roc_auc = roc_auc_metric.compute(references=labels, prediction_scores=probabilities)

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        # "roc_auc": roc_auc["roc_auc"]
    }


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Epoch {state.epoch} ended. ")


class ClassificationTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name, task_name, config=None):
        super().__init__(base_model_name=base_model_name, task_name=task_name, config=config)
        self.label_metrics = None
        self.set_label_metrics()
        self.metrics_average = 'micro'
        self.dataset_types = None
        self.data_num_dict = None
        if self.config is not None:
            if "metrics_average" in self.config:
                self.metrics_average = self.config["metrics_average"]
            if "dataset_types" in self.config:
                self.dataset_types = self.config["dataset_types"]
            if "data_num_dict" in self.config:
                self.data_num_dict = self.config["data_num_dict"]
        self.data_processor = DataProcessor(task_name=self.task_name)

    def set_label_metrics(self):
        if self.task_name in [GIBBERISH_TASK, UNSAFE_PROMPT_TASK, HALLUCINATION_TASK,
                              TOXICITY_TASK, ALL_IN_ONE_UNSAFE_CONTENTS_TASK]:
            self.label_metrics = self.compute_metrics_for_single_label_tasks
        else:
            self.label_metrics = self.compute_metrics_for_multilabel_tasks

    @staticmethod
    def compute_metrics_for_multilabel_tasks(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    def compute_metrics_for_single_label_tasks(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        probabilities = sigmoid(logits[:, 1])
        return compute_metrics(labels, predictions, probabilities, metrics_average=self.metrics_average)

    def get_pretrained_model(self, label_dicts=None, id2label=None, label2id=None):
        if self.task_name in [GIBBERISH_TASK, UNSAFE_PROMPT_TASK, HALLUCINATION_TASK, TOXICITY_TASK,
                              ALL_IN_ONE_UNSAFE_CONTENTS_TASK]:
            return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                      num_labels=len(label_dicts),
                                                                      id2label=id2label,
                                                                      label2id=label2id,
                                                                      load_in_8bit=False
                                                                      )
        elif self.task_name in [SEMANTIC_TASK, TOPIC_TASK]:
            return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                      problem_type="multi_label_classification",
                                                                      num_labels=len(label_dicts),
                                                                      id2label=id2label,
                                                                      label2id=label2id,
                                                                      load_in_8bit=False
                                                                      )

    def get_training_data(self, idx=None):
        dataset, id2labels, label2ids, label_names = self.data_processor.get_dataset(dataset_types=self.dataset_types,
                                                                                     data_num_dict=self.data_num_dict)
        write_hf_dataset_to_csv(dataset['train'], f"{self.task_name}_train_data_{idx}.csv")
        write_hf_dataset_to_csv(dataset['validation'], f"{self.task_name}_validation_data_{idx}.csv")
        print(f"dataset in training: {dataset}")
        print(f"sample data = {dataset['train'][0]}")
        write_hf_dataset_to_csv(dataset['test'], f"{self.task_name}_test_data_{idx}.csv")
        print(f"label name = {label_names}, label2id = {label2ids}, id2labels = {id2labels}")
        return dataset, label_names, id2labels, label2ids

    def get_encoded_dataset(self, dataset, tokenizer):
        encoded_dataset = self.data_processor.process_encoded_datasets(dataset=dataset, tokenizer=tokenizer)
        print(f"encoded_dataset in training: {encoded_dataset}")
        return encoded_dataset

    def evaluate(self, tokenizer=None, trainer=None):
        dataset = DataLoader().process_a_subdataset_for_all_in_one_task(dataset_type="toxic-chat")
        dataset = DatasetDict({'train': dataset})
        encoded_dataset = self.data_processor.process_encoded_datasets(dataset=dataset, tokenizer=tokenizer)
        test_results = trainer.evaluate(eval_dataset=encoded_dataset["train"])
        print("Test Results with toxic-chat data:", test_results)

    def train(self, model, encoded_dataset, batch_size=32, idx=None):
        # config_manager = TrainingConfigManager(self.task_name, self.base_model_name, config=self.config)
        model = get_peft_model(model, TrainingConfigManager.get_lora_config(model_name=self.base_model_name))
        model.print_trainable_parameters()  # see % trainable parameters
        output_dir = self.base_model_name.split("/")[-1] + "-" + self.task_name + "-" + idx
        trainer = Trainer(
            model=model,
            args=TrainingConfigManager.get_training_config(output_dir=output_dir, task_name=self.task_name,
                                                           batch_size=batch_size),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.label_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()]
        )
        trainer.train()
        test_results = trainer.evaluate(eval_dataset=encoded_dataset["test"])
        print("Test Results with hybrid test data:", test_results)
        model.save_pretrained(output_dir + "-final")
        return trainer

    def process(self, desired_total_data_n=None, batch_size=32):
        t = str(datetime.now())
        dataset, label_names, id2labels, label2ids = self.get_training_data(idx=t)
        model = self.get_pretrained_model(label_names, id2labels, label2ids)
        tokenizer = get_tokenizer(base_model_name=self.base_model_name)
        model.config.pad_token_id = model.config.eos_token_id
        encoded_dataset = self.get_encoded_dataset(dataset=dataset, tokenizer=tokenizer)
        trainer = self.train(model=model, encoded_dataset=encoded_dataset, batch_size=batch_size, idx=t)
        self.evaluate(tokenizer=tokenizer, trainer=trainer)
