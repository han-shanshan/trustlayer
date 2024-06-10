import os
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from data_operation.data_reader import DataReader
from utils.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, \
    HALLUCINATION_TASK_NAME, TOXICITY_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME, \
    HALLUCINATION_EXPLANATION_TASK_NAME, EXPLANATION_RESPONSE_TEMPLATE
from utils.file_operations import write_a_dictionary_to_file


def remove_newlines(data_entry):
    for key, value in data_entry.items():
        if isinstance(value, str):  # Check if the field is a string
            data_entry[key] = value.replace("\n", " ")  # Replace newlines with space
    return data_entry


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, task_name, dataset_types: list = None, data_num_dict=None, desired_total_data_n=None,
                  training_per=0.8, validation_per=0.1, test_per=0.1):
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
        elif task_name == HALLUCINATION_EXPLANATION_TASK_NAME:
            task_data = self.get_hybrid_hallucination_data(dataset_types, data_num_dict=data_num_dict)
            return task_data
        elif task_name == TOXICITY_TASK_NAME:
            if dataset_types is None:
                task_data = self.load_toxicity_data()  # default
            else:
                task_data = self.load_toxic_sophisticated_data()
        elif task_name == ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME:
            task_data = self.all_in_one_data(dataset_types, data_num_dict=data_num_dict)
            return task_data
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

    def get_hybrid_hallucination_data(self, dataset_types: list = None, data_num_dict=None):
        if data_num_dict is None:
            raise ValueError(f"data num dict is None!")
        print(f"dataset_types = {dataset_types}")
        dataset_list = []

        for i in range(len(dataset_types)):
            dataset_type = dataset_types[i]
            dataset = self.process_a_subdataset_for_hybrid_hallucination_data(dataset_type)
            dataset_list.append(dataset.shuffle(seed=0))

        # dataset_list = self.remove_duplicates_between_datasets(dataset_list) # todo: when there are more datasets

        test_dataset, training_dataset, validation_dataset = self.create_a_hybrid_dataset_based_on_data_num_dict(
            data_num_dict, dataset_types, dataset_list)
        
        
        training_dataset = training_dataset.map(lambda example: {
        "text": self.get_llama_prompt_for_hallucination_reasoning_task(example["input"], example["output"])})
        validation_dataset = validation_dataset.map(lambda example: {
        "text": self.get_llama_prompt_for_hallucination_reasoning_task(example["input"], example["output"])})
        test_dataset = test_dataset.map(lambda example: {
        "text": self.get_llama_prompt_for_hallucination_reasoning_task(example["input"], "")})
        
        datasets = DatasetDict({
            'train': training_dataset.remove_columns([col for col in training_dataset.column_names if col not in ["text"]]),
            'validation': validation_dataset.remove_columns([col for col in training_dataset.column_names if col not in ["text"]]),
            'test': test_dataset.remove_columns([col for col in training_dataset.column_names if col not in ["text", "output"]])
        })
        print(f"final datasets = {datasets}")
        return datasets

    @staticmethod
    def get_llama_prompt_for_hallucination_reasoning_task(input, output):
        return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> According to the Question and the Contexts, is there any hallucination in the LLM Answer?  {input}. {EXPLANATION_RESPONSE_TEMPLATE}[/INST] {output}. "

    # @staticmethod
    # def get_llama_prompt_for_hallucination_reasoning_task(input):
    #     return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> Is there hallucination in the Answer based on the Question and the Context?  {input}. [/INST] "

    def process_a_subdataset_for_hybrid_hallucination_data(self, dataset_type):
        sub_dataset = None
        # if dataset_type == "rag-hallucination1000":
        #     sub_dataset = load_dataset("neural-bridge/rag-hallucination-dataset-1000")
        #     sub_dataset = self._merge_several_datasets_of_different_phases([sub_dataset])
        #     sub_dataset = sub_dataset.map(lambda example: {"Input": f"Question: {example['question']}; Context: {example['context']}; Answer: {example['answer']}", "Output": "No, the context does not contain necessary information to answer the question. "})
        #     print(sub_dataset)
        #     sub_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates([sub_dataset], col_name1="Input", col_name2="Output")
        if dataset_type == "HaluEval":
            sub_dataset = load_dataset("pminervini/HaluEval", "qa")["data"]
            sub_dataset.filter(
                lambda example: example['question'] is not None and example["knowledge"] is not None and example[
                    "right_answer"] is not None and example["hallucinated_answer"] is not None)
            sub_dataset1 = sub_dataset.map(lambda example: {
                "input": f"Question: {example['question']}; Context: {example['knowledge']}; LLM Answer: {example['right_answer']}",
                "output": "No, the answer can be deduced from the context. "})
            sub_dataset2 = sub_dataset.map(lambda example: {
                "input": f"Question: {example['question']}; Context: {example['knowledge']}; LLM Answer: {example['hallucinated_answer']}",
                "output": f"Yes, the answer cannot be deduced from the context or the answer is useless. "})
            sub_dataset = concatenate_datasets([sub_dataset1, sub_dataset2])

        # sub_dataset = self.remove_duplicates_in_a_dataset(sub_dataset, col_name1="input", col_name2="output")
        
        return sub_dataset

    """
    (train #, evaluation #, testing #)
    Unsafe data: 
    1. HEx-PHI: https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI
       - 330 for training
    2. (?) ChatGPT-Jailbreak-Prompts: https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts

    Safe data: 
    1. Hotpot QA: https://huggingface.co/datasets/hotpot_qa
    2. TruthfulQA: https://huggingface.co/datasets/truthful_qa?row=3
    3. awesome chatgpt prompts: https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
    4. personalized prompts: https://huggingface.co/datasets/andrewsiah/filtered_personalization_prompt_response
    5. qa chat prompts: https://huggingface.co/datasets/nm-testing/qa-chat-prompts?row=0
    6. 10k_prompts_ranked: https://huggingface.co/datasets/DIBT/10k_prompts_ranked?row=34
    7. iterative-prompt: https://huggingface.co/datasets/RLHFlow/iterative-prompt-v1-iter1-20K?row=0

    Hybrid data: 
    1. toxic-chat: https://huggingface.co/datasets/lmsys/toxic-chat
    2. openai-moderation: https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation
       - 1680
    3. gibberish: https://huggingface.co/datasets/Sowmya15/March06_gibberish
    """

    def all_in_one_data(self, dataset_types: list = None, data_num_dict=None):
        if data_num_dict is None:
            raise ValueError(f"data num dict is None!")
        print(f"dataset_types = {dataset_types}")
        dataset_list = []

        for i in range(len(dataset_types)):
            dataset_type = dataset_types[i]
            dataset = self.process_a_subdataset_for_all_in_one_task(dataset_type)
            dataset_list.append(dataset.shuffle(seed=0))

        dataset_list = self.remove_duplicates_between_datasets(dataset_list)

        test_dataset, training_dataset, validation_dataset = self.create_a_hybrid_dataset_based_on_data_num_dict(
            data_num_dict, dataset_types, dataset_list)

        datasets = DatasetDict({
            'train': training_dataset,
            'validation': validation_dataset,
            'test': test_dataset
        })
        datasets = datasets.shuffle(seed=0)
        print(f"final datasets = {datasets}")
        return datasets

        # if dataset_type == "lmsys-chat":
        #     data = load_dataset("lmsys/lmsys-chat-1m")["train"]
        #     print(data)
        #     print(f"sample data = {data[0]}")

    def create_a_hybrid_dataset_based_on_data_num_dict(self, data_num_dict, dataset_types, dataset_list):
        dataset_label_counter_meta = {}
        training_dataset = None
        validation_dataset = None
        test_dataset = None
        for i in range(len(dataset_types)):
            dataset_type = dataset_types[i]
            training_data_num = data_num_dict[dataset_type]["train"]
            validation_data_num = data_num_dict[dataset_type]["validation"]
            test_data_num = data_num_dict[dataset_type]["test"]

            meta = {}
            if training_data_num > 0:
                sub_data = dataset_list[i].select(range(training_data_num))
                if training_dataset is None:
                    training_dataset = sub_data
                else:
                    training_dataset = concatenate_datasets([training_dataset, sub_data])
                if "label" in sub_data.column_names:
                    meta["training_label_1"], meta["training_label_0"], meta[
                        "training_total"] = self.count_label_numbers(
                        sub_data)
            if validation_data_num > 0:
                sub_data = dataset_list[i].select(range(training_data_num, training_data_num + validation_data_num))
                if validation_dataset is None:
                    validation_dataset = sub_data
                else:
                    validation_dataset = concatenate_datasets([validation_dataset, sub_data])
                if "label" in sub_data.column_names:
                    meta["validation_label_1"], meta["validation_label_0"], meta[
                        "validation_total"] = self.count_label_numbers(sub_data)
            if test_data_num > 0:
                sub_data = dataset_list[i].select(range(training_data_num + validation_data_num,
                                                        training_data_num + validation_data_num + test_data_num))
                if test_dataset is None:
                    test_dataset = sub_data
                else:
                    test_dataset = concatenate_datasets([test_dataset, sub_data])
                if "label" in sub_data.column_names:
                    meta["test_label_1"], meta["test_label_0"], meta["test_total"] = self.count_label_numbers(sub_data)
            dataset_label_counter_meta[dataset_type] = meta
        write_a_dictionary_to_file(file_name="../training/dataset_label_counter_meta.txt",
                                   dictionary=dataset_label_counter_meta)
        return test_dataset, training_dataset, validation_dataset

    @staticmethod
    def remove_duplicates_between_datasets(dataset_list):
        seen_texts = set()
        for i in range(len(dataset_list)):
            D = dataset_list[i]
            temp_duplicate_set = set()
            for _, example in enumerate(D):
                if example['text'].strip() in seen_texts:
                    temp_duplicate_set.add(example['text'].strip())
                else:
                    seen_texts.add(example['text'].strip())
            dataset_list[i] = D.filter(lambda e: e['text'].strip() not in temp_duplicate_set)
        return dataset_list

    def process_a_subdataset_for_all_in_one_task(self, dataset_type):
        sub_dataset = None
        if dataset_type == "HEx-PHI":
            sub_dataset = self.load_HEx_PHI_data()
        elif dataset_type == "toxic-chat":  # toxic-chat0124 is better than toxic-chat1123
            sub_dataset = self.process_toxic_chat_data(load_dataset("lmsys/toxic-chat", "toxicchat0124"),
                                                       remove_jailbreaking=False)
            sub_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates([sub_dataset])
            # the quality is hard to evaluate
        elif dataset_type == "openai":  # this dataset has duplicates; should be removed with str.strip()
            sub_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
            sub_dataset = sub_dataset.map(lambda example: {'text': example['prompt'],
                                                           'label': 1 if any([example[col] == 1 for
                                                                              col in
                                                                              ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2',
                                                                               'V']]) else 0})
        elif dataset_type in ["hotpot_qa", "truthful_qa"]:
            if dataset_type == "hotpot_qa":
                sub_dataset = load_dataset("hotpot_qa", 'distractor')["train"]  # ['distractor', 'fullwiki']
            else:
                sub_dataset = load_dataset("truthful_qa", "generation")["validation"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["question"], "label": 0})
        elif dataset_type == "mt-bench":
            sub_dataset = load_dataset("lmsys/mt_bench_human_judgments")["human"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["conversation_a"][0]["content"], "label": 0})
        elif dataset_type == "awesome_chatgpt_prompts":
            sub_dataset = load_dataset("fka/awesome-chatgpt-prompts")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["prompt"], "label": 0})
        elif dataset_type in ["jigsaw", "jigsaw-toxic-only"]:
            sub_dataset = self.load_toxic_sophisticated_data(desired_number=150000)
            if dataset_type == "jigsaw-toxic-only":
                sub_dataset = sub_dataset.filter(lambda example: example["label"] == 1)
        elif dataset_type == "gibberish":
            sub_dataset = self.filter_non_records(load_dataset("Sowmya15/March06_gibberish"), "text")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"label": 1 if example["label"] != 0 else 0})
            sub_dataset = sub_dataset.filter(lambda example: example["label"] == 1)
        elif dataset_type == "jailbreak":
            sub_dataset = load_dataset("jackhhao/jailbreak-classification")["train"]
            sub_dataset = sub_dataset.map(
                lambda example: {"text": example["prompt"], "label": 1 if example["type"] == "jailbreak" else 0})
            sub_dataset = sub_dataset.filter(lambda example: example["label"] == 1)
        elif dataset_type == "gpt-jailbreak":
            sub_dataset = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")["train"]
            sub_dataset = sub_dataset.map(
                lambda example: {"text": example["Prompt"], "label": 1})
        elif dataset_type == "personalization_prompt":
            sub_dataset = load_dataset("andrewsiah/filtered_personalization_prompt_response")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["prompt"], "label": 0})
        elif dataset_type == "qa-chat-prompts":
            sub_dataset = load_dataset("nm-testing/qa-chat-prompts")["train_sft"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["prompt"], "label": 0})
        elif dataset_type == "chatgpt-prompts":
            sub_dataset = load_dataset("MohamedRashad/ChatGPT-prompts")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["human_prompt"], "label": 0})
        elif dataset_type == "10k_prompts_ranked":
            sub_dataset = load_dataset("DIBT/10k_prompts_ranked")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["prompt"], "label": 0})
        elif dataset_type == "iterative-prompt":
            sub_dataset = load_dataset("RLHFlow/iterative-prompt-v1-iter1-20K")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["context"].replace("Here is a request of a "
                                                                                              "user for an AI "
                                                                                              "assistant. User:",
                                                                                              '').strip(), "label": 0})
        elif dataset_type == "instruction-following":
            sub_dataset = load_dataset("wis-k/instruction-following-eval")["train"]
            sub_dataset = sub_dataset.map(lambda example: {"text": example["prompt"], "label": 0})
        print(f"dataset name = {dataset_type}, {sub_dataset}")
        sub_dataset = self.drop_duplicates_in_a_dataset(sub_dataset, col_name="text")
        sub_dataset = sub_dataset.remove_columns(
            [col for col in sub_dataset.column_names if col not in ["text", "label"]])
        print(f"{dataset_type}: {sub_dataset}")
        print(f"sample data = {sub_dataset[0]}\n=====================\n")
        sub_dataset = sub_dataset.filter(lambda example: example['label'] is not None and example["text"] is not None)
        return sub_dataset

    @staticmethod
    def count_label_numbers(sub_dataset):
        label_one_num = len(sub_dataset.filter(lambda example: example['label'] == 1))
        label_zero_num = len(sub_dataset.filter(lambda example: example['label'] == 0))
        return label_one_num, label_zero_num, len(sub_dataset)

    @staticmethod
    def drop_duplicates_in_a_dataset(dataset, col_name="text"):
        df = dataset.to_pandas()
        df_unique = df.drop_duplicates(subset=[col_name])  # Drop duplicates based on 'text' column
        dataset = Dataset.from_pandas(df_unique)
        return dataset

    @staticmethod
    def print_distinct_column_name(dataset, column_name):
        df = dataset.to_pandas()

        # Get distinct values from the column
        distinct_values = df[column_name].unique()
        print(f"distinct values in column {column_name}: {distinct_values}")

    @staticmethod
    def load_HEx_PHI_data():
        data_frames = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_file_directory = os.path.join(dir_path, '..', 'cache', 'downloaded_data', 'HEx-PHI')
        for filename in os.listdir(csv_file_directory):
            if filename.endswith('.csv'):  # Check for CSV files
                file_path = os.path.join(csv_file_directory, filename)
                df = pd.read_csv(file_path, header=None, names=['text'])
                data_frames.append(df)
        merged_df = pd.concat(data_frames, ignore_index=True)  # Concatenate all dataframes into one
        # self.detect_duplicates_in_pd_dataset(merged_df)
        hf_dataset = Dataset.from_pandas(merged_df)
        hf_dataset = hf_dataset.map(lambda example: {'text': example['text'], 'label': 1})
        return hf_dataset

    @staticmethod
    def detect_duplicates_in_pd_dataset(merged_df):
        duplicates = merged_df[merged_df.duplicated('text', keep=False)]
        if not duplicates.empty:
            print("Duplicates found:")
            print(duplicates)
        else:
            print("No duplicates found.")

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
        toxic_chat_data_subset2 = self.process_toxic_chat_data(load_dataset("lmsys/toxic-chat", "toxicchat0124"))

        merged_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates(
            [jigsaw_comment_dataset, jigsaw_unindended_bias_data,
             toxic_chat_data_subset2, toxicity3M_dataset])
        return merged_dataset

    def load_toxic_sophisticated_data(self, desired_number=None):
        bias_data = self._load_original_jigsaw_unindended_bias_dataset()
        bias_data = bias_data.remove_columns([item for item in bias_data["train"].column_names
                                              if
                                              item not in ['toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
                                                           'identity_attack', 'insult', 'threat', 'sexual_explicit',
                                                           'toxicity_annotator_count',
                                                           'comment_text']])
        merged_dataset = self._merge_several_datasets_of_different_phases([bias_data])

        filtered_jigsaw_data = self._jissaw_data_process_duplicate_texts_and_numeric_values_in_record(merged_dataset)
        dataset = Dataset.from_pandas(filtered_jigsaw_data).rename_column('comment_text', 'text')
        dataset = self.filter_a_split_of_hf_dataset(dataset, "text")
        if desired_number is not None:
            if not isinstance(desired_number, int):
                raise ValueError(f"undesired value of desired_number: {desired_number}")
            dataset = dataset.select(range(desired_number))

        def create_label_based_on_columns(example):
            sum_score = example['toxicity'] + example['severe_toxicity'] + example['obscene'] + example[
                'sexual_explicit'] \
                        + example['identity_attack'] + example['insult'] + example['threat']
            example['label'] = 0
            if (sum_score >= 0.5 and example['toxicity_annotator_count'] > 20) \
                    or (0.3 < sum_score < 0.5 and 30 <= example['toxicity_annotator_count'] <= 100) \
                    or 50 < example['toxicity_annotator_count'] <= 100 \
                    or (sum_score >= 0.5 and ('Trump' in example['text'])):
                example['label'] = 1
            return example

        dataset = dataset.map(create_label_based_on_columns)
        return dataset

    @staticmethod
    def _jissaw_data_process_duplicate_texts_and_numeric_values_in_record(merged_dataset, key_column="comment_text"):
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

    def merge_datasets_of_different_phases_and_remove_duplicates(self, dataset_list: list,
                                                                 col_name1="text", col_name2="label"):
        merged_dataset = self._merge_several_datasets_of_different_phases(dataset_list)
        merged_datasets_without_duplicates = self.remove_duplicates_in_a_dataset(merged_dataset, col_name2, col_name1)
        return merged_datasets_without_duplicates

    @staticmethod
    def remove_duplicates_in_a_dataset(dataset, col_name2, col_name1):
        dataset_dicts = dataset.to_dict()
        unique_texts = {}
        for i, (col1_value, col2_value) in enumerate(zip(dataset_dicts[col_name1], dataset_dicts[col_name2])):
            if col1_value is not None:
                col1_value = col1_value.strip()
                if col1_value in unique_texts:
                    if unique_texts[col1_value] is not None and unique_texts[col1_value] != col2_value:
                        unique_texts[col1_value] = None  # mark conflicts
                        print(f"conflict founded ---------")
                else:
                    unique_texts[col1_value] = col2_value
        transformed_data = {
            col_name1: list(unique_texts.keys()),
            col_name2: list(unique_texts.values())
        }
        merged_datasets_without_duplicates = Dataset.from_dict(transformed_data)
        merged_datasets_without_duplicates = merged_datasets_without_duplicates.filter(
            lambda example: example[col_name2] is not None)
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
        dataset = DataReader.read_csv_file_data(csv_file_path=csv_file_path)  # 1999516
        return dataset
        # if desired_number is None:
        #     return dataset
        # if not isinstance(desired_number, int):
        #     raise ValueError(f"undesired value of desired_number: {desired_number}")
        # return dataset["train"].select(range(desired_number))

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
    def process_toxic_chat_data(toxic_chat_data, remove_jailbreaking=True):
        print(f"original dataset = {toxic_chat_data}")
        from langdetect import detect
        import re

        def remove_jailbreaking_and_non_english_inputs(example):
            return example['human_annotation'] and example["jailbreaking"] == 0 and \
                   not re.compile(
                       r'^(\d+[-+*/]\d+ = \d+;\s*)*(\d+(\s*[-+*/]\s*\d+)+ = \?|(\d+\s*[-+*/]\s*\d+\s*\?)|(\d+\s*['
                       r'-+*/]\s*\d+\s*=\s*))$').match(
                       example["user_input"]) and detect(example["user_input"]) == 'en'

        def remove_non_english_inputs(example):
            return example['human_annotation'] and not re.compile(
                r'^(\d+[-+*/]\d+ = \d+;\s*)*(\d+(\s*[-+*/]\s*\d+)+ = \?|(\d+\s*[-+*/]\s*\d+\s*\?)|(\d+\s*['
                r'-+*/]\s*\d+\s*=\s*))$').match(example["user_input"]) and detect(example["user_input"]) == 'en'

        if remove_jailbreaking:
            filter_function = remove_jailbreaking_and_non_english_inputs
        else:
            filter_function = remove_non_english_inputs
        toxic_chat_data = toxic_chat_data.filter(filter_function)

        if remove_jailbreaking:
            toxic_chat_data = toxic_chat_data.remove_columns(
                [col for col in toxic_chat_data["train"].column_names if col not in ["user_input", "toxicity"]]
            )
            for split in toxic_chat_data.keys():
                toxic_chat_data[split] = toxic_chat_data[split].rename_column("toxicity", "label")
                toxic_chat_data[split] = toxic_chat_data[split].rename_column("user_input", "text")
            return toxic_chat_data
        else:
            toxic_chat_data = toxic_chat_data.map(lambda example: {'text': example['user_input'],
                                                                   'label': 0 if (example['toxicity'] == 0 and example[
                                                                       'jailbreaking'] == 0) else 1})
            toxic_chat_data = toxic_chat_data.remove_columns(
                [col for col in toxic_chat_data["train"].column_names if
                 col not in ["text", "label"]]
            )

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

# if __name__ == '__main__':
# sub_dataset = load_dataset("jackhhao/jailbreak-classification")["train"]
# sub_dataset = sub_dataset.map(
#     lambda example: {"text": example["prompt"], "label": 1 if example["type"] == "jailbreak" else 0})
# sub_dataset = sub_dataset.filter(lambda example: example["label"] == 1)
# print(sub_dataset)
# data_types = [
#     # "rag-hallucination1000", # 1000 in total
#     "HaluEval"
#               ]
# data_num_dict = {
#     "HaluEval": {"train": 8000, "validation": 1500, "test": 500},
#     # "rag-hallucination1000": {"train": 500, "validation": 20, "test": 0},
# }
# data = DataLoader().get_hybrid_hallucination_data(dataset_types=data_types, data_num_dict=data_num_dict)


# dataset_types = [
#     # "mt-bench", "HEx-PHI",  # "toxic-chat",
#     #              "openai", "hotpot_qa",
#     #              "truthful_qa",
#     #              "awesome_chatgpt_prompts", "jigsaw",
#     #              #  "gibberish",
#     #              "gpt-jailbreak", "jailbreak",
#     # "personalization_prompt", "qa-chat-prompts",
#     # "chatgpt-prompts", "10k_prompts_ranked",
#     # "iterative-prompt"
# ]
#
# data_num_dict = {
#     "HEx-PHI": {"train": 330, "validation": 0, "test": 0},
#     # "toxic-chat": {"train": 0, "validation": 200, "test": 0},
#     "openai": {"train": 160, "validation": 1500, "test": 0},
#     "hotpot_qa": {"train": 500, "validation": 200, "test": 200},
#     "truthful_qa": {"train": 500, "validation": 100, "test": 100},
#     "awesome_chatgpt_prompts": {"train": 0, "validation": 150, "test": 0},
#     "jigsaw": {"train": 50000, "validation": 2000, "test": 300},
#     # "gibberish": {"train": 1000, "validation": 150, "test": 100},
#     "mt-bench": {"train": 0, "validation": 80, "test": 0},
#     "gpt-jailbreak": {"train": 0, "validation": 78, "test": 0},
#     "jailbreak": {"train": 400, "validation": 0, "test": 70},
#     "personalization_prompt": {"train": 1000, "validation": 800, "test": 200},
#     "qa-chat-prompts": {"train": 0, "validation": 200, "test": 0},
#     "chatgpt-prompts": {"train": 360, "validation": 0, "test": 0},
#     "10k_prompts_ranked": {"train": 5000, "validation": 2000, "test": 500},
#     "iterative-prompt": {"train": 5000, "validation": 2000, "test": 500},
# }
# dataloader = DataLoader()
# dataset = dataloader.all_in_one_data(dataset_types, data_num_dict=data_num_dict)
# print(dataset)
#
# training_data_df = dataset["train"].to_pandas()
# dataloader.detect_duplicates_in_pd_dataset(training_data_df)
# validation_df = dataset["validation"].to_pandas()
# dataloader.detect_duplicates_in_pd_dataset(validation_df)
# test_df = dataset["test"].to_pandas()
# dataloader.detect_duplicates_in_pd_dataset(test_df)
#
# print(f"original dataset = {dataset}")
#
# new_dataset = dataloader.merge_datasets_of_different_phases_and_remove_duplicates([dataset])
# print(f"new dataset = {new_dataset}")
# d = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
# print(d)
