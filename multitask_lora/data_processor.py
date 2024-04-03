from transformers import AutoTokenizer, AutoConfig
import numpy as np
from datasets import concatenate_datasets, load_dataset, DatasetDict

from multitask_lora.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME


class DataProcessor():
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
        elif self.task_name is SEMANTIC_TASK_NAME:
            return "train", "test", "validation"
        else:
            Exception("wrong task name")

    def get_remove_column_names(self, dataset):
        train_phase_name, _, _ = self.get_phase_names()
        return dataset[train_phase_name].column_names

    def prepare_label_dict_for_a_task(self, dataset):
        if self.task_name is SEMANTIC_TASK_NAME:
            return [label_name for label_name in dataset['train'].features.keys() if label_name not in ['ID', 'Tweet']]
        elif self.task_name is TOPIC_TASK_NAME:
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

            for label_list in dataset['train_2021']['label_name']:
                label_names.update(label_list)

            return list(label_names)

    @staticmethod
    def load_data(task_name):
        if task_name == TOPIC_TASK_NAME:
            return load_dataset("cardiffnlp/tweet_topic_multi")
        elif task_name == SEMANTIC_TASK_NAME:
            return load_dataset("sem_eval_2018_task_1", "subtask5.english")
        else:
            return None

    # task_name == SEMANTIC_TASK_NAME:

    def get_encoded_datasets(self):
        dataset = self.load_data(task_name=self.task_name)  # load_dataset("cardiffnlp/tweet_topic_multi")
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
        encoded_dataset = dataset.map(self.process_data, batched=True,
                                      remove_columns=self.get_remove_column_names(dataset))
        if self.task_name == TOPIC_TASK_NAME:
            encoded_dataset = self.concatenate_dataset_of_same_phase(encoded_dataset)

        final_dataset = DatasetDict(encoded_dataset)
        # d2label[idx] for idx, label in enumerate(encoded_dataset['train'][0]['labels']) if label == 1.0]}")
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
                concatenated_dataset[phase] = concatenate_datasets([concatenated_dataset[phase], encoded_dataset[original_phase_name]])
            else:
                concatenated_dataset[phase] = encoded_dataset[original_phase_name]
        return concatenated_dataset

    def process_data(self, examples):
        if self.task_name == SEMANTIC_TASK_NAME:
            return self.process_semantics_data(examples)
        elif self.task_name is TOPIC_TASK_NAME:
            return self.process_topic_data(examples)

    def process_topic_data(self, examples):
        text = examples['text']

        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # print(f"labels = {self.labels[TOPIC_TASK_NAME]}")
        # ############# check if any bugs in the following code in two versions:: rewrite the following line for topic task
        # # the datasets have different structures
        # labels_batch = {k: examples[k] for k in examples['label_name'] if k in self.labels[TOPIC_TASK_NAME]}
        # # print(f"label_batch = {examples.keys()}")
        # print(f"examples['label_name'] = {examples['label_name']}")
        # print(f"label batch = {labels_batch}")

        labels_batch = {}
        # print(f"self.labels[TOPIC_TASK_NAME]={self.labels[TOPIC_TASK_NAME]}")
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

        # print(f"label_batch = {labels_batch}")
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding
