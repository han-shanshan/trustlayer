import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import re
import os

"""
Reference code: 
https://huggingface.co/docs/transformers/main/en/peft
https://github.com/huggingface/peft/discussions/661
"""


class InferenceEngine:
    def __init__(self, default_task):
        self.task_name = default_task
        general_config_file_path = '../inference_config.json'

        with open(general_config_file_path, 'r') as file:
            self.config = json.load(file)
        print(f"config = {self.config}")
        self.base_model_name = self.config['base_model_name_or_path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def set_task(self, task_type):
        self.task_name = task_type

    def get_checkpoint_directory(self, checkpoint_id=None):
        directory = self.base_model_name.split("/")[1] + "-" + self.task_name
        final_model_path = directory + "-final"
        if os.path.isdir(final_model_path):
            entries = os.listdir(final_model_path)  # List all entries in the directory
            if len(entries) > 0:
                return final_model_path
        if checkpoint_id is not None and checkpoint_id > 0:
            return directory + "/checkpoint-" + str(checkpoint_id)
        max_checkpoint_id = -1
        pattern = re.compile(r'^(.*?)-(\d+)$')
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                num = int(match.group(2))
                if num > max_checkpoint_id:
                    max_checkpoint_id = num
        return directory + "/checkpoint-" + str(max_checkpoint_id)

    def inference(self, text, checkpoint_id=None):
        path = self.get_checkpoint_directory(checkpoint_id)
        print(f"model path = {path}")
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                        problem_type="multi_label_classification",
                                                                        num_labels=len(self.config[self.task_name]),
                                                                        id2label=self.config[self.task_name]
                                                                        )
        base_model.load_adapter(path)
        encoding = self.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(base_model.device) for k, v in encoding.items()}
        outputs = base_model(**encoding)
        logits = outputs.logits

        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [self.config[self.task_name][str(idx)] for idx, label in enumerate(predictions) if label == 1.0]
        return predicted_labels
