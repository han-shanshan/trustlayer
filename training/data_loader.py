import os
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from data_operation.data_reader import DataReader
from training.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, \
    HALLUCINATION_TASK_NAME, TOXICITY_TASK_NAME


def remove_newlines(data_entry):
    for key, value in data_entry.items():
        if isinstance(value, str):  # Check if the field is a string
            data_entry[key] = value.replace("\n", " ")  # Replace newlines with space
    return data_entry


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, task_name, dataset_type=None, desired_total_data_n=None, training_per=0.8, validation_per=0.1,
                  test_per=0.1):
        # None: return full dataset by default
        if task_name == TOPIC_TASK_NAME:
            task_data = load_dataset("cardiffnlp/tweet_topic_multi")
        elif task_name == SEMANTIC_TASK_NAME:
            task_data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        elif task_name == GIBBERISH_TASK_NAME:
            task_data = self.load_gibberish_data()
        elif task_name == UNSAFE_PROMPT_TASK_NAME:
            task_data = self.load_unsafe_prompt_data()
        elif task_name == HALLUCINATION_TASK_NAME:
            task_data = self.load_hallucination_data()
        elif task_name == TOXICITY_TASK_NAME:
            if dataset_type is None:
                task_data = self.load_toxicity_data()  # default
            else:
                task_data = self.load_toxic_sophisticated_data()
        else:
            task_data = None
        print(f"-----task name = {task_name}------\n original dataset: {task_data}")

        # downloaded_dataset = downloaded_dataset.shuffle(seed=0)
        split_dataset = self.shuffle_a_dataset_and_get_splitted(task_data, test_per, validation_per)
        print(f"split data = {split_dataset}")

        if desired_total_data_n is None:  # return full dataset by default
            return split_dataset
        small_dataset = self.get_a_small_dataset(split_dataset, desired_total_data_n, test_per, training_per,
                                                 validation_per)
        print(f"new dataset: {DatasetDict(small_dataset)}")
        return small_dataset

    @staticmethod
    def get_a_small_dataset(downloaded_dataset, desired_total_data_num, test_per, training_per,
                            validation_per):  # for testing use
        for k in downloaded_dataset.keys():
            if "train" in k and int(desired_total_data_num * training_per) < len(downloaded_dataset[k]):
                downloaded_dataset[k] = downloaded_dataset[k].select(range(int(desired_total_data_num * training_per)))
            if "validation" in k and int(desired_total_data_num * validation_per) < len(downloaded_dataset[k]):
                downloaded_dataset[k] = downloaded_dataset[k].select(
                    range(int(desired_total_data_num * validation_per)))
            if "test" in k and int(desired_total_data_num * test_per) < len(downloaded_dataset[k]):
                downloaded_dataset[k] = downloaded_dataset[k].select(range(int(desired_total_data_num * test_per)))
        return DatasetDict(downloaded_dataset)

    @staticmethod
    def load_unsafe_prompt_data():
        dataset = load_dataset("deepset/prompt-injections")
        test_validation_split = dataset["test"].train_test_split(test_size=0.5)
        dataset["validation"] = test_validation_split["train"]
        dataset["test"] = test_validation_split["test"]
        return DatasetDict(dataset)

    def load_toxicity_data(self):
        # # this error should have been fixed? TypeError: TextEncodeInput must be Union[TextInputSequence,
        # Tuple[InputSequence, InputSequence]]
        jigsaw_comment_dataset = self._load_jigsaw_comment_dataset()
        jigsaw_unindended_bias_data = self._load_jigsaw_unindended_bias_dataset()
        toxicity3M_dataset = self._load_toxicity_data_3M()
        toxic_chat_data_subset1 = self._process_toxic_chat_subdata(load_dataset("lmsys/toxic-chat", "toxicchat1123"))
        toxic_chat_data_subset2 = self._process_toxic_chat_subdata(load_dataset("lmsys/toxic-chat", "toxicchat0124"))

        merged_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates(
            [jigsaw_comment_dataset, jigsaw_unindended_bias_data,
             toxic_chat_data_subset1, toxic_chat_data_subset2,
             toxicity3M_dataset])
        return merged_dataset

    def load_toxic_sophisticated_data(self):
        bias_data = self._load_original_jigsaw_unindended_bias_dataset()
        bias_data = bias_data.remove_columns([item for item in bias_data["train"].column_names
                                              if
                                              item not in ['toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
                                                           'identity_attack', 'insult', 'threat', 'sexual_explicit',
                                                           'toxicity_annotator_count',
                                                           'comment_text']])
        merged_dataset = self._merge_several_datasets_of_different_phases([bias_data])
        filtered_jigsawdata = self._jissaw_data_process_duplicate_texts_and_numeric_values_in_record(merged_dataset)
        dataset = Dataset.from_pandas(filtered_jigsawdata).rename_column('comment_text', 'text')
        dataset = self.filter_a_split_of_hf_dataset(dataset, "text")

        def create_label_based_on_columns(example):
            sum_score = example['toxicity'] + example['severe_toxicity'] + example['obscene'] + example['sexual_explicit'] \
                  + example['identity_attack'] + example['insult'] + example['threat']
            example['label'] = 0
            if (sum_score >= 0.5 and example['toxicity_annotator_count'] > 20) \
                    or (0.3 < sum_score < 0.5 and 30 <= example['toxicity_annotator_count'] <= 100) \
                    or 50 < example['toxicity_annotator_count'] <= 100\
                    or (sum_score >= 0.5 and ('Trump' in example['text'])):
                example['label'] = 1
            return example

        dataset = dataset.map(create_label_based_on_columns)
        return dataset

    @staticmethod
    def _jissaw_data_process_duplicate_texts_and_numeric_values_in_record(merged_dataset, key_column ="comment_text"):
        df = merged_dataset.to_pandas()
        df = df.drop_duplicates()
        duplicate_keys = df[df.duplicated(subset=[key_column], keep=False)]
        grouped = duplicate_keys.groupby(key_column)
        non_duplicate_keys_df = df.drop_duplicates(subset=[key_column], keep=False)
        merged_records = []
        deleted_record_counter = 0
        for key, group in grouped:
            avg_record = {key_column: key}
            for column in group.columns:
                if column != key_column:
                    unique_values = group[column].unique()
                    if len(unique_values) == 1:
                        avg_record[column] = unique_values[0]  # all values are the same, keep the original value
                    else:
                        # values are different, compute the average for numeric columns
                        if np.issubdtype(group[column].dtype, np.number):
                            avg_record[column] = group[column].mean()
                        else:
                            print("errors!")  # merged_record[column] = ', '.join(map(str, unique_values))
                else:
                    deleted_record_counter += len(group[column])
            merged_records.append(avg_record)
        avg_records_df = pd.DataFrame(merged_records)
        return pd.concat([non_duplicate_keys_df, avg_records_df], ignore_index=True)

    def merge_datasets_of_different_phases_and_remove_duplicates(self, dataset_list: list):
        merged_dataset = self._merge_several_datasets_of_different_phases(dataset_list)
        dataset_dicts = merged_dataset.to_dict()
        unique_texts = {}
        for i, (text, label) in enumerate(zip(dataset_dicts["text"], dataset_dicts["label"])):
            if text is not None:
                text = text.strip()
                if text in unique_texts:
                    if unique_texts[text] is not None and unique_texts[text] != label:
                        unique_texts[text] = None  # mark conflicts
                        # print(f"conflict founded ---------")
                else:
                    unique_texts[text] = label

        transformed_data = {
            "text": list(unique_texts.keys()),
            "label": list(unique_texts.values())
        }
        merged_datasets_without_duplicates = Dataset.from_dict(transformed_data)
        merged_datasets_without_duplicates = merged_datasets_without_duplicates.filter(
            lambda example: example['label'] is not None)
        return merged_datasets_without_duplicates

    @staticmethod
    def _merge_several_datasets_of_different_phases(dataset_list):
        merged_dataset = None
        for D in dataset_list:
            for split in D.keys():
                if merged_dataset is None:
                    merged_dataset = D[split]
                else:
                    merged_dataset = concatenate_datasets([merged_dataset, D[split]])
        # print(f"merge data: {merged_dataset}")

        return merged_dataset

    @staticmethod
    def shuffle_a_dataset_and_get_splitted(merged_dataset, test_per, validation_per):
        merged_dataset = merged_dataset.shuffle(seed=0)
        train_testval_split = merged_dataset.train_test_split(test_size=test_per + validation_per)
        train_dataset = train_testval_split['train']
        testval_dataset = train_testval_split['test']
        test_validation_split = testval_dataset.train_test_split(test_size=test_per / (test_per + validation_per))
        test_dataset = test_validation_split['test']
        validation_dataset = test_validation_split['train']
        split_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'validation': validation_dataset
        })
        return split_dataset

    @staticmethod
    def _load_jigsaw_comment_dataset():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        """
            DatasetDict({
                train: Dataset({
                    features: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
                    num_rows: 159571
                })
            })
            """
        csv_file_path = os.path.join(dir_path, '..', 'cache', 'downloaded_data',
                                     'jigsaw-toxic-comment-classification-challenge', 'train.csv')
        jigsaw_comment_dataset = DataReader.read_csv_file_data(csv_file_path=csv_file_path)

        def create_label_for_jigsaw_comment_dataset(example):
            example['label'] = 1 if any([example[col] == 1 for col in
                                         ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                                          'identity_hate']]) else 0
            return example

        jigsaw_comment_dataset['train'] = jigsaw_comment_dataset['train'].map(create_label_for_jigsaw_comment_dataset)
        jigsaw_comment_dataset['train'] = jigsaw_comment_dataset['train'].remove_columns(
            ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
        jigsaw_comment_dataset = jigsaw_comment_dataset.rename_column('comment_text', 'text')
        return jigsaw_comment_dataset

    @staticmethod
    def _load_original_jigsaw_unindended_bias_dataset():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        """
                DatasetDict({
                    train: Dataset({
                        features: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
                        num_rows: 159571
                    })
                })
                """
        csv_file_path = os.path.join(dir_path, '..', 'cache', 'downloaded_data',
                                     'jigsaw-unintended-bias-in-toxicity-classification', 'all_data.csv')
        return DataReader.read_csv_file_data(csv_file_path=csv_file_path)  # 1999516

    def _load_jigsaw_unindended_bias_dataset(self):
        jigsaw_unindended_bias_data = self._load_original_jigsaw_unindended_bias_dataset()
        # ######### testing ##################
        # def filter_toxicity_annotator_count(example):
        #     return (30 < example['toxicity_annotator_count'] < 50) and all(
        #         example[col] < 0.5 for col in [
        #             'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat'
        #         ]
        #     )
        # filtered_dataset = jigsaw_unindended_bias_data.filter(filter_toxicity_annotator_count)
        # ######################################
        def create_label_for_jigsaw_unindended_bias_dataset(example):
            example['label'] = 1 if (any([example[col] > 0.5 for col in
                                          ['toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
                                           'identity_attack', 'insult', 'threat']])
                                     or example['toxicity_annotator_count'] > 30
                                     ) else 0
            return example

        jigsaw_unindended_bias_data = jigsaw_unindended_bias_data.map(create_label_for_jigsaw_unindended_bias_dataset)
        jigsaw_unindended_bias_data = jigsaw_unindended_bias_data.rename_column('comment_text', 'text')
        jigsaw_unindended_bias_data = jigsaw_unindended_bias_data.remove_columns([item for item in
                                                                                  jigsaw_unindended_bias_data[
                                                                                      "train"].column_names if
                                                                                  item not in ["label", "text"]])
        return jigsaw_unindended_bias_data

    def _load_toxicity_data_3M(self):
        toxicity_data_3m = load_dataset("FredZhang7/toxi-text-3M")
        toxicity_data_3m = toxicity_data_3m.filter(lambda example: example['lang'] == 'en')
        for split in toxicity_data_3m.keys():
            toxicity_data_3m[split] = toxicity_data_3m[split].rename_column("is_toxic", "label")
        filtered_data = self.filter_non_records(toxicity_data_3m, "text")
        print(f"filtered_data = {filtered_data}")
        return filtered_data.remove_columns(["lang"])

    @staticmethod
    def _process_toxic_chat_subdata(toxic_chat_data):
        from langdetect import detect
        import re

        def remove_jailbreaking_and_non_english_inputs(example):
            return example["jailbreaking"] == 0 and \
                   not re.compile(
                       r'^(\d+[-+*/]\d+ = \d+;\s*)*(\d+(\s*[-+*/]\s*\d+)+ = \?|(\d+\s*[-+*/]\s*\d+\s*\?)|(\d+\s*['
                       r'-+*/]\s*\d+\s*=\s*))$').match(
                       example["user_input"]) \
                   and detect(example["user_input"]) == 'en'

        toxic_chat_data = toxic_chat_data.filter(remove_jailbreaking_and_non_english_inputs)
        toxic_chat_data = toxic_chat_data.remove_columns(
            [col for col in toxic_chat_data["train"].column_names if col not in ["user_input", "toxicity"]]
        )
        for split in toxic_chat_data.keys():
            toxic_chat_data[split] = toxic_chat_data[split].rename_column("toxicity", "label")
            toxic_chat_data[split] = toxic_chat_data[split].rename_column("user_input", "text")
        return toxic_chat_data

    @staticmethod
    def load_hallucination_data():
        dataset = load_dataset("cemuluoglakci/hallucination_evaluation")
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column("answer_label_id", "label")
        test_validation_split = dataset["train"].train_test_split(test_size=0.2)
        test_validation_split = test_validation_split["test"].train_test_split(test_size=0.5)
        dataset["validation"] = test_validation_split["train"]
        dataset["test"] = test_validation_split["test"]
        return dataset

    def load_gibberish_data(self):
        # "0": "clean",    "1": "noise",     "2": "word salad"
        gibberish_dataset_names = ["Sowmya15/March06_gibberish",  # tag: 0, 1, 2
                                   "Sowmya15/gibberish_april2",  # tag: 0, 1
                                   "Sowmya15/March11_gibberish",  # tag: 0, 1
                                   ]
        gibberish_datasets = []
        # Dataset "Sowmya15/gibberish_march22" has to be loaded manually; tag: 0, 1, 2;
        gibberish_march22_dataset = self._load_gibberish_mar22_manually()
        gibberish_datasets.append(gibberish_march22_dataset)
        gibberish_datasets.append(self._load_multilingual_gibberish_data())

        for dataset_name in gibberish_dataset_names:
            data = self.filter_non_records(load_dataset(dataset_name), "text")
            print(f"{dataset_name} dataset: {data}; \n sample data for {dataset_name}: {data['train'][0]}")
            gibberish_datasets.append(data)

        non_gibberish_data = self._load_data_and_set_fixed_lables(dataset_id="iohadrubin/not-gibberish-20-56-22",
                                                                  label_value=0, removed_column=['file_loc', 'id'])
        # not quite sure about the following dataset; the text is too long  # todo: may remove in later training
        word_salad_data = self._load_data_and_set_fixed_lables(dataset_id="iohadrubin/gibberish-47-15-08",
                                                               label_value=2, removed_column=['file_loc', 'id'])

        print(f"non_gibberish_data example data: {non_gibberish_data['train'][0]}")
        print(f"non_gibberish_data: {non_gibberish_data}")
        print(f"word_salad_data example data: {word_salad_data['train'][0]}")
        print(f"word_salad_data: {word_salad_data}")
        gibberish_datasets.append(non_gibberish_data)
        gibberish_datasets.append(word_salad_data)

        merged_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates(gibberish_datasets)
        return merged_dataset

    @staticmethod
    def _load_multilingual_gibberish_data():
        multilingual_gibberish_data = load_dataset("Sowmya15/profanity_multi_march25")
        print(f"original multilingual data = {multilingual_gibberish_data}")
        multilingual_gibberish_data = multilingual_gibberish_data.filter(
            lambda example: example['language'] == "English")
        multilingual_gibberish_data = multilingual_gibberish_data.remove_columns(["language"])
        print(f"multilingual data after filtering and removing cols= {multilingual_gibberish_data}")
        return multilingual_gibberish_data

    @staticmethod
    def _load_data_and_set_fixed_lables(dataset_id, label_value, removed_column: list = None):
        data = load_dataset(dataset_id)

        def add_label(example):
            example['label'] = label_value
            return example

        data = data.map(add_label)
        return data.remove_columns(removed_column)

    def _load_gibberish_mar22_manually(self):
        gibberish_mar22_training_dataset = self._read_gibberish_march22_from_csv(file_name='train.csv')['train']

        def remove_unwanted_column(example):
            if 'Unnamed: 0' in example:
                del example['Unnamed: 0']
            return example

        gibberish_mar22_training_dataset = gibberish_mar22_training_dataset.map(remove_unwanted_column, batched=True)
        gibberish_march22_dataset = DatasetDict({
            'train': gibberish_mar22_training_dataset,
            'test': self._read_gibberish_march22_from_csv(file_name='test.csv')['train'],
            'validation': self._read_gibberish_march22_from_csv(file_name='validation.csv')['train']
        })
        gibberish_march22_dataset = self.filter_non_records(gibberish_march22_dataset, "text")
        print(f"gibberish_march22_dataset={gibberish_march22_dataset}")
        print(f"gibberish_march22_dataset sample={gibberish_march22_dataset['train'][0]}")
        return gibberish_march22_dataset

    @staticmethod
    def _read_gibberish_march22_from_csv(file_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_file_path = os.path.join(dir_path, '..', 'cache', 'downloaded_data',
                                     'gibberish_mar22', file_name)
        return DataReader.read_csv_file_data(csv_file_path=csv_file_path)

    def filter_non_records(self, dataset, col_name):
        filtered_dataset = {}
        for phase in dataset.keys():
            filtered_dataset_in_phase = self.filter_a_split_of_hf_dataset(dataset[phase], col_name)
            filtered_dataset[phase] = filtered_dataset_in_phase
        # def change_labels(example):
        #     example['label'] = np.argmax(example['label'])
        # filtered_dataset = DatasetDict(filtered_dataset).map(change_labels)
        return DatasetDict(filtered_dataset)

    @staticmethod
    def filter_a_split_of_hf_dataset(dataset_phase, col_name):
        return dataset_phase.filter(lambda example: example[col_name] is not None)
