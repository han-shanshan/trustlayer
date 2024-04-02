from transformers import AutoTokenizer, AutoConfig
import numpy as np
from constants import SEMANTIC_TASK_NAME, TOPIC_TASK_NAME

class DataProcessor():
    def __init__(self, tokenizer=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer
        self.labels = None
        self.task_name = None

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
        
    def process_data_diff_datasets(self, examples): 
        if self.task_name == SEMANTIC_TASK_NAME:
            return self.process_semantics_data_diff_datasets(examples)
        elif self.task_name is TOPIC_TASK_NAME:
            return self.process_topic_data_diff_datasets(examples)
        

    def process_topic_data_diff_datasets(self, examples):
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
    
    def process_semantics_data_diff_datasets(self, examples):
        # take a batch of texts and encode them
        text = examples["Tweet"] 
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels[SEMANTIC_TASK_NAME]}

        # print(f"label_batch = {labels_batch}")
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels[SEMANTIC_TASK_NAME])))
        # fill numpy array
        for idx, label in enumerate(self.labels[SEMANTIC_TASK_NAME]):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding


    def process_data_unified_datasets(self, examples): 
        if self.task_name == SEMANTIC_TASK_NAME:
            return self.process_semantics_data_unified_datasets(examples)
        elif self.task_name is TOPIC_TASK_NAME:
            return self.process_topic_data_unified_datasets(examples)

    def process_topic_data_unified_datasets(self, examples):
        text = examples['text']
        # print(f"self.labels = {self.labels}")
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
        # labels_batch = {k: examples[k] for k in examples['label_name'] if k in self.labels}
        # print(f"label_batch = {examples.keys()}")
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

        
    def process_semantics_data_unified_datasets(self, examples):
        # take a batch of texts and encode them
        text = examples["Tweet"] 
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}

        for l in self.labels:
            if l not in labels_batch:
                labels_batch[l] = [False for _ in range(len(text))]

        # print(f"label_batch = {labels_batch}")
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding
