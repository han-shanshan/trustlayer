import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from data_operation.data_reader import DataReader
from training.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, \
    HALLUCINATION_TASK_NAME, TOXICITY_TASK_NAME


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, task_name, desired_total_data_n=None, training_per=0.8, validation_per=0.1, test_per=0.1):
        # None: return full dataset by default
        if task_name == TOPIC_TASK_NAME:
            downloaded_dataset = load_dataset("cardiffnlp/tweet_topic_multi")
        elif task_name == SEMANTIC_TASK_NAME:
            downloaded_dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        elif task_name == GIBBERISH_TASK_NAME:
            downloaded_dataset = self.load_gibberish_data()
        elif task_name == UNSAFE_PROMPT_TASK_NAME:
            downloaded_dataset = self.load_unsafe_prompt_data()
        elif task_name == HALLUCINATION_TASK_NAME:
            downloaded_dataset = self.load_hallucination_data()
        elif task_name == TOXICITY_TASK_NAME:
            downloaded_dataset = self.load_toxicity_data(training_per=0.8, validation_per=0.1, test_per=0.1)
        else:
            downloaded_dataset = None
        print(f"-----task name = {task_name}------\n original dataset: {downloaded_dataset}")
        # downloaded_dataset = downloaded_dataset.shuffle(seed=0)
        return downloaded_dataset

        # if desired_total_data_n is None:  # return full dataset by default
        #     return downloaded_dataset
        # small_dataset = self.get_a_small_dataset(downloaded_dataset, desired_total_data_n, test_per, training_per,
        #                                          validation_per)
        # print(f"new dataset: {DatasetDict(small_dataset)}")
        # return small_dataset

    # @staticmethod
    # def get_a_small_dataset(downloaded_dataset, desired_total_data_num, test_per, training_per, validation_per):
    #     for k in downloaded_dataset.keys():
    #         if "train" in k:
    #             if int(desired_total_data_num * training_per) < len(downloaded_dataset[k]):
    #                 downloaded_dataset[k] = downloaded_dataset[k].select(
    #                     range(int(desired_total_data_num * training_per)))
    #         if "validation" in k:
    #             if int(desired_total_data_num * validation_per) < len(downloaded_dataset[k]):
    #                 downloaded_dataset[k] = downloaded_dataset[k].select(
    #                     range(int(desired_total_data_num * validation_per)))
    #         if "test" in k:
    #             if int(desired_total_data_num * test_per) < len(downloaded_dataset[k]):
    #                 downloaded_dataset[k] = downloaded_dataset[k].select(range(int(desired_total_data_num * test_per)))
    #     return DatasetDict(downloaded_dataset)

    @staticmethod
    def load_unsafe_prompt_data():
        dataset = load_dataset("deepset/prompt-injections")
        test_validation_split = dataset["test"].train_test_split(test_size=0.5)
        dataset["validation"] = test_validation_split["train"]
        dataset["test"] = test_validation_split["test"]
        return DatasetDict(dataset)

    def load_toxicity_data(self, training_per=0.8, validation_per=0.1, test_per=0.1):
        toxicity_data_3m = self._load_toxicity_data_3M()
        # TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
        jigsaw_comment_dataset = self._load_jigsaw_comment_dataset()
        jigsaw_unindended_bias_data = self._load_jigsaw_unindended_bias_dataset()
        toxicity3M_dataset = self._load_toxicity_data_3M()
        toxic_chat_data_subset1 = self._process_toxic_chat_subdata(load_dataset("lmsys/toxic-chat", "toxicchat1123"))
        toxic_chat_data_subset2 = self._process_toxic_chat_subdata(load_dataset("lmsys/toxic-chat", "toxicchat0124"))
        merged_dataset = None
        for D in [toxicity_data_3m, jigsaw_comment_dataset, jigsaw_unindended_bias_data, toxicity3M_dataset,
                  toxic_chat_data_subset1, toxic_chat_data_subset2]:
            for split in D.keys():
                if merged_dataset is None:
                    merged_dataset = D[split]
                else:
                    merged_dataset = concatenate_datasets([merged_dataset, D[split]])
        merged_dataset = merged_dataset.shuffle(seed=0)
        print(f"{type(merged_dataset)} {merged_dataset}")

        return merged_dataset

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
        print(jigsaw_comment_dataset)
        print(jigsaw_comment_dataset["train"][0])
        return jigsaw_comment_dataset

    @staticmethod
    def _load_jigsaw_unindended_bias_dataset():
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
        jigsaw_unindended_bias_data = DataReader.read_csv_file_data(csv_file_path=csv_file_path)  # 1999516
        print(jigsaw_unindended_bias_data)
        print(jigsaw_unindended_bias_data["train"][0])

        # ######### testing ##################
        # def filter_toxicity_annotator_count(example):
        #     return (30 < example['toxicity_annotator_count'] < 50) and all(
        #         example[col] < 0.5 for col in [
        #             'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat'
        #         ]
        #     )
        # filtered_dataset = jigsaw_unindended_bias_data.filter(filter_toxicity_annotator_count)
        # ######################################

        def create_label_for_jigsaw_jigsaw_unindended_bias_dataset(example):
            example['label'] = 1 if (any([example[col] > 0.5 for col in
                                          ['toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
                                           'identity_attack', 'insult', 'threat']])
                                     or example['toxicity_annotator_count'] > 30
                                     ) else 0
            return example

        jigsaw_unindended_bias_data = jigsaw_unindended_bias_data.map(
            create_label_for_jigsaw_jigsaw_unindended_bias_dataset)
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
        return filtered_data.remove_columns(["lang"])

    @staticmethod
    def _process_toxic_chat_subdata(toxic_chat_data):
        from langdetect import detect

        def remove_jailbreaking_and_non_english_inputs(example):
            return example["jailbreaking"] == 0 and detect(example["user_input"]) == 'en'

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
        # dataset = load_dataset("imdb")
        dataset = load_dataset("Sowmya15/March06_gibberish")
        return self.filter_non_records(dataset, "text")

    @staticmethod
    def filter_non_records(dataset, col_name):
        filtered_dataset = {}
        for phase in dataset.keys():
            filtered_dataset_in_phase = dataset[phase].filter(lambda example: example[col_name] is not None)
            filtered_dataset[phase] = filtered_dataset_in_phase
            print(f"{phase}: {len(dataset[phase])} ==== {len(filtered_dataset[phase])}")
        #
        # def change_labels(example):
        #     example['label'] = np.argmax(example['label'])
        # filtered_dataset = DatasetDict(filtered_dataset).map(change_labels)
        return DatasetDict(filtered_dataset)


if __name__ == '__main__':
    # desired_total_data_n = 10000
    # dataset = DataLoader().load_data(TOXICITY_TASK_NAME, desired_total_data_n=None)
    DataLoader().load_toxicity_data()
