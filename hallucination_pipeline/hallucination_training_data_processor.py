import numpy as np
from datasets import DatasetDict, Dataset
from multitask_lora.constants import CUSTOMIZED_HALLUCINATION_TASK_NAME
from multitask_lora.data_processor import DataProcessor


class HallucinationTrainingDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__(task_name=CUSTOMIZED_HALLUCINATION_TASK_NAME)

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_phase_names(self):
        return "train", "test", "validation"

    def get_remove_column_names(self, dataset):
        remove_columns = dataset["train"].column_names.append("question_and_knowledge")
        return remove_columns

    def prepare_label_dict_for_a_task(self, df):
        labels = df['explain'].unique()
        print("Distinct values in Column explain:", labels)
        return labels

    # def encoding(self, dataset):
    #     return dataset.map(self.process_customized_hallucination_data, batched=True,
    #                        remove_columns=self.get_remove_column_names(dataset))

    def process_encoded_datasets(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        encoded_dataset = dataset.map(self.process_customized_hallucination_data, batched=True,
                           remove_columns=self.get_remove_column_names(dataset))
        final_dataset = DatasetDict(encoded_dataset)
        final_dataset.set_format("torch")
        return final_dataset

    def get_dataset_info(self, desired_total_data_n=None, file_path=None, training_per=0.8, validation_per=0.1,
                         test_per=0.1):
        import pandas as pd
        df = pd.read_excel(file_path)
        # todo: translate to English
        connecting_phrase = " Please make inference based on the following knowledge: "
        df['question_and_knowledge'] = df['question'] + connecting_phrase + df['knowledge']
        print(df)
        label_names = self.prepare_label_dict_for_a_task(df)
        print(f"label names = {label_names}")
        idx = 0
        id2labels = {}
        label2ids = {}
        for label in label_names:
            id2labels[idx] = label
            label2ids[label] = idx
            idx += 1
        self.set_labels(labels=label_names)

        dataset = Dataset.from_pandas(df)
        dataset_dict = DatasetDict({
            'train': dataset.shuffle(seed=42).select(range(int(training_per * len(dataset)))),
            'test': dataset.shuffle(seed=42).select(
                range(int(training_per * len(dataset)), int((training_per + test_per) * len(dataset)))),
            'validation': dataset.shuffle(seed=42).select(
                range(int((training_per + test_per) * len(dataset)), len(dataset)))
        })

        print(dataset_dict)
        return dataset, id2labels, label2ids, label_names

    def process_customized_hallucination_data(self, examples):
        text = examples["question_and_knowledge"]
        encoding = self.tokenizer(text, text_pair=examples["response"], padding="max_length", #truncation=True,
                                  # max_length=512,
                                  return_tensors="pt")
        labels_batch = {}
        for label in self.labels:
            labels_batch[label] = []
        for label_names_of_one_record in examples['explain']:
            for label in self.labels:
                if label in label_names_of_one_record:
                    labels_batch[label].append(True)
                else:
                    labels_batch[label].append(False)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding
