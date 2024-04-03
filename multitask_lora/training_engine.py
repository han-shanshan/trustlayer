import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from multitask_lora.bert.data_processor import DataProcessor


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


class TrainingEngine:
    def __init__(self, base_model_name, task_name, training_args=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.training_args = training_args

    def set_task_type(self, task_name):
        self.task_name = task_name

    @staticmethod
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        data_processor = DataProcessor(tokenizer=tokenizer, task_name=self.task_name)
        encoded_dataset, id2label, label2id, label_dicts = data_processor.get_encoded_datasets()


        model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
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
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        print("=======print_trainable_parameters============")
        model.print_trainable_parameters()  # see % trainable parameters
        # training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=500)
        batch_size = 8
        output_dir = self.base_model_name.split("/")[1] + "-" + self.task_name
        if self.training_args is None:
            self.training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=1,
                weight_decay=0.01,
                load_best_model_at_end=True,
                # metric_for_best_model=metric_name,
                # push_to_hub=True,
            )

        bert_peft_trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.compute_metrics,
        )
        bert_peft_trainer.train()
        model.save_pretrained(output_dir + "-final")
