from datasets import load_dataset, DatasetDict
from multitask_lora.constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME, GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, \
    HALLUCINATION_TASK_NAME, TOXICITY_TASK_NAME


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, task_name, desired_total_data_n=None, training_per=0.8, validation_per=0.1, test_per=0.1):
        # None: return full dataset by default
        if task_name == TOPIC_TASK_NAME:
            dataset = load_dataset("cardiffnlp/tweet_topic_multi")
        elif task_name == SEMANTIC_TASK_NAME:
            dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        elif task_name == GIBBERISH_TASK_NAME:
            dataset = self.load_gibberish_data()
        elif task_name == UNSAFE_PROMPT_TASK_NAME:
            dataset = self.load_unsafe_prompt_data()
        elif task_name == HALLUCINATION_TASK_NAME:
            dataset = self.load_hallucination_data()
        elif task_name == TOXICITY_TASK_NAME:
            dataset = self.load_toxicity_data()
        else:
            dataset = None
        print(f"-----task name = {task_name}------\n original dataset: {dataset}")
        dataset = dataset.shuffle(seed=0)

        if desired_total_data_n is None:  # return full dataset by default
            return dataset
        small_dataset = self.get_a_small_dataset(dataset, desired_total_data_n, test_per, training_per, validation_per)
        print(f"new dataset: {DatasetDict(small_dataset)}")
        return small_dataset

    @staticmethod
    def get_a_small_dataset(dataset, desired_total_data_num, test_per, training_per, validation_per):
        for k in dataset.keys():
            if "train" in k:
                if int(desired_total_data_num * training_per) < len(dataset[k]):
                    dataset[k] = dataset[k].select(range(int(desired_total_data_num * training_per)))
            if "validation" in k:
                if int(desired_total_data_num * validation_per) < len(dataset[k]):
                    dataset[k] = dataset[k].select(range(int(desired_total_data_num * validation_per)))
            if "test" in k:
                if int(desired_total_data_num * test_per) < len(dataset[k]):
                    dataset[k] = dataset[k].select(range(int(desired_total_data_num * test_per)))
        return DatasetDict(dataset)

    @staticmethod
    def load_unsafe_prompt_data():
        dataset = load_dataset("deepset/prompt-injections")
        test_validation_split = dataset["test"].train_test_split(test_size=0.5)
        dataset["validation"] = test_validation_split["train"]
        dataset["test"] = test_validation_split["test"]
        return DatasetDict(dataset)

    def load_toxicity_data(self):
        dataset = load_dataset("FredZhang7/toxi-text-3M")
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column("is_toxic", "label")
        # TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
        return self.filter_non_records(dataset, "text")

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
    desired_total_data_n = 10000
    DataLoader().load_data(TOXICITY_TASK_NAME, desired_total_data_n=desired_total_data_n)
    DataLoader().load_data(SEMANTIC_TASK_NAME, desired_total_data_n=desired_total_data_n)
    DataLoader().load_data(GIBBERISH_TASK_NAME, desired_total_data_n=desired_total_data_n)
    DataLoader().load_data(UNSAFE_PROMPT_TASK_NAME, desired_total_data_n=desired_total_data_n)
    DataLoader().load_data(HALLUCINATION_TASK_NAME, desired_total_data_n=desired_total_data_n)
    DataLoader().load_data(TOPIC_TASK_NAME, desired_total_data_n=desired_total_data_n)

