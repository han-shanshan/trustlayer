import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit as sigmoid
from training.training_engine import compute_metrics
from utils.constants import GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, TOXICITY_TASK_NAME, \
    HALLUCINATION_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME, \
    FOX_BASE_GPU
import torch
import json
import os

"""
Reference code: 
https://huggingface.co/docs/transformers/main/en/peft
https://github.com/huggingface/peft/discussions/661
"""


class InferenceEngine:
    def __init__(self, default_task, base_model, adapter_path=None, inference_config=None, problem_type=None):
        self.task_name = default_task
        if inference_config is not None:
            self.config = inference_config
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            general_config_file_path = os.path.join(dir_path, 'inference_config.json')
            with open(general_config_file_path, 'r') as file:
                self.config = json.load(file)
        print(f"config = {self.config}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if problem_type is not None and problem_type in ["single_label_classification", "multi_label_classification"]:
            self.problem_type = problem_type
        elif self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, TOXICITY_TASK_NAME,
                                HALLUCINATION_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME]:
            self.problem_type = "single_label_classification"
        else:
            self.problem_type = "multi_label_classification"
        if base_model == FOX_BASE_GPU:
            self.model_name = "Fox"
        else:
            self.model_name = base_model.split("/")[1]
        print(f"base_model = {base_model}")
        if adapter_path is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(adapter_path,
                                                                                 problem_type=self.problem_type,
                                                                                 num_labels=len(
                                                                                     self.config[self.task_name]),
                                                                                 id2label=self.config[self.task_name]
                                                                                 )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                                                 problem_type=self.problem_type,
                                                                                 num_labels=len(self.config[self.task_name]),
                                                                                 id2label=self.config[self.task_name]
                                                                                 )

    def set_task(self, task_type):
        self.task_name = task_type

    # def get_checkpoint_directory(self):
    #     directory = "../model/adapters/" + self.model_name + "-" + self.task_name
    #     final_model_path = directory + "-final"
    #     if os.path.isdir(final_model_path):
    #         entries = os.listdir(final_model_path)  # List all entries in the directory
    #         if len(entries) > 0:
    #             return final_model_path
    #     max_checkpoint_id = -1
    #     pattern = re.compile(r'^(.*?)-(\d+)$')
    #     for filename in os.listdir(directory):
    #         match = pattern.match(filename)
    #         if match:
    #             num = int(match.group(2))
    #             if num > max_checkpoint_id:
    #                 max_checkpoint_id = num
    #     return directory + "/checkpoint-" + str(max_checkpoint_id)

    """
    https://huggingface.co/docs/peft/en/quicktour
    """
    def inference(self, text):
        if isinstance(text, list):
            encoding = self.tokenizer(text[0], text_pair=text[1], padding=True, truncation=True, return_tensors="pt")
        else:
            encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        # encoding = self.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits

        if self.problem_type == "multi_label_classification":
            # apply sigmoid + threshold
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1
            # turn predicted id's into actual label names
            predicted_label = [self.config[self.task_name][str(idx)] for idx, label in enumerate(predictions) if
                               label == 1.0]
        else:
            predicted_label_idx = np.argmax(logits, axis=1).item()
            print(f"logits = {logits}, label = {predicted_label_idx}")
            predicted_label = self.config[self.task_name][str(predicted_label_idx)]

        return predicted_label

    def evaluation(self, texts: list, labels: list, pair_texts: list = None):
        predictions = []
        probabilities = []
        if pair_texts is None:
            for i in range(len(texts)):
                encoding = self.tokenizer(texts[i], padding=True, truncation=True, return_tensors="pt")
                encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
                outputs = self.model(**encoding)
                logits = outputs.logits
                predicted_label_idx = torch.argmax(logits, dim=-1).item()
                probability = sigmoid(logits[:, 1]).item()
                predictions.append(predicted_label_idx)
                probabilities.append(probability)

            metrics = compute_metrics(labels, predictions, probabilities)
            print(f"metrics = {metrics}")
