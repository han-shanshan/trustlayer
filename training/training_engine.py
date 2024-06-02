import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainerCallback, \
    TrainerState, TrainerControl
from transformers import Trainer
from peft import get_peft_model
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from training.training_config_manager import TrainingConfigManager
from utils.constants import GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME, \
    TOXICITY_TASK_NAME, MODEL_NAME_TINYLAMMA, FOX_BASE_GPU, SEMANTIC_TASK_NAME, TOPIC_TASK_NAME, \
    CUSTOMIZED_HALLUCINATION_TASK_NAME, HALLUCINATION_REASONING_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME
from data_operation.data_processor import DataProcessor
import evaluate
from utils.file_operations import write_hf_dataset_to_csv
from scipy.special import expit as sigmoid

# accuracy = evaluate.load("accuracy")
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")
roc_auc_metric = evaluate.load("roc_auc")
# roc_auc_metric = load_metric("roc_auc")


def compute_metrics(labels, predictions, probabilities, metrics_average="macro"):
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    roc_auc = roc_auc_metric.compute(references=labels, prediction_scores=probabilities)

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        "roc_auc": roc_auc["roc_auc"]
    }


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Epoch {state.epoch} ended. Custom logic here.")


class TrainingEngine:
    def __init__(self, base_model_name, task_name, config=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.label_metrics = None
        self.set_label_metrics()
        self.config = config
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

    def set_task_type(self, task_name):
        self.task_name = task_name

    def set_label_metrics(self):
        if self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME,
                              TOXICITY_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME]:
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

    def get_pretrained_model(self, label_dicts, id2label, label2id):
        if self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME, TOXICITY_TASK_NAME,
                              CUSTOMIZED_HALLUCINATION_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME]:
            return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                      num_labels=len(label_dicts),
                                                                      id2label=id2label,
                                                                      label2id=label2id,
                                                                      load_in_8bit=False
                                                                      )
        elif self.task_name in [SEMANTIC_TASK_NAME, TOPIC_TASK_NAME]:
            return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                      problem_type="multi_label_classification",
                                                                      num_labels=len(label_dicts),
                                                                      id2label=id2label,
                                                                      label2id=label2id,
                                                                      load_in_8bit=False
                                                                      )
        elif self.task_name in [HALLUCINATION_REASONING_TASK_NAME]:  # add explanations for inference results
            pass

    def get_tokenizer(self, model):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.base_model_name in [MODEL_NAME_TINYLAMMA, FOX_BASE_GPU]:
            # tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.padding_side = 'right'  # to prevent warnings
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        return tokenizer

    def train(self, desired_total_data_n=None):
        data_processor = DataProcessor(task_name=self.task_name)
        dataset, id2labels, label2ids, label_names = data_processor.get_dataset(dataset_types=self.dataset_types,
                                                                                data_num_dict=self.data_num_dict,
                                                                                desired_total_data_n=desired_total_data_n)
        print(f"sample data = {dataset['train'][0]}")
        write_hf_dataset_to_csv(dataset['test'], f"{self.task_name}_test_data.csv")
        model = self.get_pretrained_model(label_names, id2labels, label2ids)

        print(f"label name = {label_names}, label2id = {label2ids}, id2labels = {id2labels}")
        tokenizer = self.get_tokenizer(model)
        encoded_dataset = data_processor.process_encoded_datasets(dataset=dataset, tokenizer=tokenizer)

        config_manager = TrainingConfigManager(self.task_name, self.base_model_name, config=self.config)
        print("=======start loading metric=========")
        # metric = evaluate.load("accuracy")
        # Define LoRA Config
        model = get_peft_model(model, config_manager.get_lora_config())
        print("=======print_trainable_parameters============")
        model.print_trainable_parameters()  # see % trainable parameters
        # training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=500)
        output_dir = self.base_model_name.split("/")[-1] + "-" + self.task_name

        peft_trainer = Trainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=8),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.label_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()]
        )

        peft_trainer.train()
        test_results = peft_trainer.evaluate(eval_dataset=encoded_dataset["test"])
        print("Test Results:", test_results)
        model.save_pretrained(output_dir + "-final")

