from transformers import AutoTokenizer, AutoConfig
import numpy as np
from datasets import concatenate_datasets, load_dataset, DatasetDict
from multitask_lora.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME
from multitask_lora.data_loader import DataLoader


class DataProcessor:
    def __init__(self, tokenizer=None, task_name=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer
        self.labels = None
        self.task_name = task_name

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_phase_names(self):
        if self.task_name is TOPIC_TASK_NAME:
            return "train_2020", "test_2020", "validation_2020"
        elif self.task_name in [SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME]:
            return "train", "test", "validation"
        else:
            Exception("wrong task name")

    def get_remove_column_names(self, dataset):
        if self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME]:
            return "text"
        train_phase_name, _, _ = self.get_phase_names()
        # print(f"dataset[train_phase_name].column_names = {dataset[train_phase_name].column_names}")
        return dataset[train_phase_name].column_names

    def prepare_label_dict_for_a_task(self, dataset):
        if self.task_name is SEMANTIC_TASK_NAME:
            return [label_name for label_name in dataset['train'].features.keys() if label_name not in ['ID', 'Tweet']]
        elif self.task_name is GIBBERISH_TASK_NAME:
            """labels: noise, word salad, clean; reference: https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457
            """
            return ['clean', 'noise', 'word salad']
        elif self.task_name is UNSAFE_PROMPT_TASK_NAME:
            return ['safe', 'unsafe']
        elif self.task_name is TOPIC_TASK_NAME: # todo: check the dataset in details
            """ 
            DatasetDict({
                test_2020: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 573})
                test_2021: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 1679})
                train_2020: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 4585})
                train_2021: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 1505})
                train_all: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 6090})
                validation_2020: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 573})
                validation_2021: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 188})
                train_random: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 4564})
                validation_random: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 573})
                test_coling2022_random: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 5536})
                train_coling2022_random: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 5731})
                test_coling2022: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 5536})
                train_coling2022: Dataset({ features: ['text', 'date', 'label', 'label_name', 'id'], num_rows: 5731})
            }) 
            """
            label_names = set()
            for label_list in dataset['train_all']['label_name']:
                label_names.update(label_list)
            return list(label_names)

    def encoding(self, dataset):
        if self.task_name is SEMANTIC_TASK_NAME:
            return dataset.map(self.process_semantics_data, batched=True,
                               remove_columns=self.get_remove_column_names(dataset))
        elif self.task_name is TOPIC_TASK_NAME:
            return dataset.map(self.process_topic_data, batched=True,
                               remove_columns=self.get_remove_column_names(dataset))
        elif self.task_name in [GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME]:
            return dataset.map(self.process_single_label_classification_data, batched=True,
                               remove_columns=self.get_remove_column_names(dataset))

    def process_encoded_datasets(self):
        dataset = DataLoader().load_data(task_name=self.task_name)
        label_names = self.prepare_label_dict_for_a_task(dataset)
        print(f"label names = {label_names}")

        idx = 0
        id2labels = {}
        label2ids = {}
        for label in label_names:
            id2labels[idx] = label
            label2ids[label] = idx
            idx += 1

        self.set_labels(labels=label_names)
        encoded_dataset = self.encoding(dataset)

        if self.task_name == TOPIC_TASK_NAME: # todo: move to data loader
            encoded_dataset = self.concatenate_dataset_of_same_phase(encoded_dataset)

        final_dataset = DatasetDict(encoded_dataset)
        final_dataset.set_format("torch")
        return final_dataset, id2labels, label2ids, label_names

    @staticmethod
    def concatenate_dataset_of_same_phase(encoded_dataset):
        concatenated_dataset = {}
        for phase in encoded_dataset.keys():  # Check if the phase already exists in D
            original_phase_name = phase
            if phase not in ["train", "test", "validation"]:
                # in case in some datasets the phases are called different names, e.g., "train_2020"
                for p in ["train", "test", "validation"]:
                    if p in phase:
                        phase = p
                        break
            if phase in concatenated_dataset:
                # Concatenate the new dataset with the existing one
                concatenated_dataset[phase] = concatenate_datasets(
                    [concatenated_dataset[phase], encoded_dataset[original_phase_name]])
            else:
                concatenated_dataset[phase] = encoded_dataset[original_phase_name]
        return concatenated_dataset

    def process_topic_data(self, examples):
        text = examples['text']
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        labels_batch = {}
        for label in self.labels:
            labels_batch[label] = []
        for label_names_of_one_record in examples['label_name']:
            for label in self.labels:
                if label in label_names_of_one_record:
                    labels_batch[label].append(True)
                else:
                    labels_batch[label].append(False)
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def process_semantics_data(self, examples):
        # take a batch of texts and encode them
        text = examples["Tweet"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def process_single_label_classification_data(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128,
                              return_tensors="pt")
