import numpy as np
from transformers import AutoModelForSequenceClassification
from scipy.special import expit as sigmoid
# from training.training_engine import compute_metrics, get_tokenizer
import torch
import json
import os
from training.classification_training_engine import compute_metrics
from utils.constants import MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE, SINGLE_LABEL_CLASSIFICATION_PROBLEM_TYPE, \
    HALLUCINATION_REASONING_TASK
from utils.util import get_tokenizer

"""
Reference code: 
https://huggingface.co/docs/transformers/main/en/peft
https://github.com/huggingface/peft/discussions/661
"""


class InferenceEngine:
    def __init__(self, task_name, base_model, model=None, adapter_path=None, inference_config=None,
                 problem_type="single_label_classification"):
        self.task_name = task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.config = self.get_inference_config(inference_config)
        print(f"config = {self.config}")
        if problem_type not in [SINGLE_LABEL_CLASSIFICATION_PROBLEM_TYPE,
                                MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE,
                                HALLUCINATION_REASONING_TASK]:
            raise Exception(f"Invalid problem_type: {problem_type}")
        self.problem_type = problem_type
        # self.model_name = base_model.split("/")[1]

        self.model = self.get_model(base_model=base_model, adapter_path=adapter_path, model=model)
        self.tokenizer = get_tokenizer(base_model_name=base_model)
        self.model.to(self.device)

    @staticmethod
    def get_inference_config(inference_config):
        if inference_config is not None:
            return inference_config
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            general_config_file_path = os.path.join(dir_path, 'inference_config.json')
            with open(general_config_file_path, 'r') as file:
                return json.load(file)

    def get_model(self, base_model=None, adapter_path=None, model=None):
        if model is not None:
            return model
        if adapter_path is not None:
            model_path = adapter_path
        elif base_model is not None:
            model_path = base_model
        else:
            raise ValueError("Base_model is None and adapter_path is None")
        return AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                  problem_type=self.problem_type,
                                                                  num_labels=len(self.config[self.task_name]),
                                                                  id2label=self.config[self.task_name],
                                                                  trust_remote_code=True)

    """
    https://huggingface.co/docs/peft/en/quicktour
    """

    def inference(self, text, text_pair=None):
        if text_pair is not None:
            encoding = self.tokenizer(text, text_pair=text_pair, padding=True, truncation=True, return_tensors="pt")
        else:
            encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        # encoding = self.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits
        if self.problem_type == MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE:
            # apply sigmoid + threshold
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1
            # turn predicted id's into actual label names
            predicted_label = [self.config[self.task_name][str(idx)] for idx, label in enumerate(predictions) if
                               label == 1.0]
        else:
            predicted_label_idx = np.argmax(logits.cpu(), axis=1).item()
            predicted_label = self.config[self.task_name][str(predicted_label_idx)]
        return predicted_label

    def evaluate(self, dataset):
        labels = dataset["label"]
        texts = dataset["text"]
        # dataset = dataset.remove_columns('label')
        predictions = []
        probabilities = []
        counter = 0
        for text in texts:
            encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=516,
                                      return_tensors="pt")
            encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
            # print(f"encoding =========================== {encoding}")
            outputs = self.model(**encoding)
            logits = outputs.logits
            predicted_label_idx = torch.argmax(logits, dim=-1).item()
            probability = sigmoid(logits[:, 1].cpu().detach()).item()
            predictions.append(predicted_label_idx)
            # print(f"predicted_label_idx = {predicted_label_idx}")
            probabilities.append(probability)

            if counter % 100 == 0:
                print(f"label = {predicted_label_idx}, real label = {labels[counter]}, text = {text}")
            counter += 1

        metrics = compute_metrics(labels, predictions, probabilities)
        print(f"metrics = {metrics}")
