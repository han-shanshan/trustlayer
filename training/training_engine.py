import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from peft import get_peft_model
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from training.training_config_manager import TrainingConfigManager
from training.constants import GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME, \
    TOXICITY_TASK_NAME, MODEL_NAME_TINYLAMMA, FOX_BASE_GPU, SEMANTIC_TASK_NAME, TOPIC_TASK_NAME, \
    CUSTOMIZED_HALLUCINATION_TASK_NAME, HALLUCINATION_REASONING_TASK_NAME
from training.data_processor import DataProcessor
import evaluate

accuracy = evaluate.load("accuracy")


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


class TrainingEngine:
    def __init__(self, base_model_name, task_name, config=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.label_metrics = None
        self.set_label_metrics()
        self.config = config

    def set_task_type(self, task_name):
        self.task_name = task_name

    def set_label_metrics(self):
        if self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME,
                              TOXICITY_TASK_NAME]:
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

    @staticmethod
    def compute_metrics_for_single_label_tasks(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    def get_pretrained_model(self, label_dicts, id2label, label2id):
        if self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME,
                              TOXICITY_TASK_NAME, CUSTOMIZED_HALLUCINATION_TASK_NAME]:
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

    def train(self):
        data_processor = DataProcessor(task_name=self.task_name)
        dataset, id2labels, label2ids, label_names = data_processor.get_dataset()
        model = self.get_pretrained_model(label_names, id2labels, label2ids)
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
        output_dir = self.base_model_name.split("/")[1] + "-" + self.task_name

        bert_peft_trainer = Trainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=8),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.label_metrics,
        )
        bert_peft_trainer.train()
        model.save_pretrained(output_dir + "-final")
