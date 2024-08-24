from datasets import load_dataset
from data_operation.data_file_operator import FileOperator
from utils.file_operations import write_hf_dataset_to_csv
from utils.util import merge_datasets_on_column_names


class FixingDataPreparer:
    def __init__(self):
        self.data_reader = FileOperator()

    def process_reasoning_data_for_hallucination(self, dataset_type):
        subset_name = ""
        dataset_name = ""
        if "-" in dataset_type:
            dataset_name = dataset_type.split("-")[0]
            subset_name = dataset_type.split("-")[1]
        dataset = None
        if dataset_type in ["HaluEval-qa", "HaluEval-dialogue", "HaluEval-summarization"]:
            dataset = load_dataset("pminervini/HaluEval", subset_name)["data"]
            print(f"dataset11111 = {dataset}")
            split_token = "Question:"
            # dataset = dataset.select(range(5))  # for testing
            if subset_name == "dialogue":
                dataset = dataset.rename_column('dialogue_history', 'question')
                dataset = dataset.rename_column('right_response', 'right_answer')
                dataset = dataset.rename_column('hallucinated_response', 'hallucinated_answer')
                split_token = "Dialogue:"
            if subset_name == "summarization":
                dataset = dataset.map(
                    lambda example: {"question": "You are a helpful assistant, please help summarize the document."})
                dataset = dataset.rename_column('right_summary', 'right_answer')
                dataset = dataset.rename_column('hallucinated_summary', 'hallucinated_answer')
                dataset = dataset.rename_column('document', 'knowledge')

            dataset.filter(lambda example: example['question'] is not None and example["knowledge"] is not None
                                           and example["right_answer"] is not None and example[
                                               "hallucinated_answer"] is not None)
            folder_name = dataset_type.split("-")[0]
            data_file_name = dataset_type.split("-")[1] + "_shuffled.csv"
            reasoning_dataset = self.data_reader.read_dataset_from_csv_file('..', 'cache', 'downloaded_data',
                                                                            folder_name, data_file_name)['train']
            reasoning_dataset = reasoning_dataset.filter(lambda example: example['is_hallucination'] == 'Yes')
            reasoning_dataset = reasoning_dataset.map(
                lambda example: {'question': example['input'].split("\nKnowledge: ")[0].split(split_token)[1].strip(),
                                 'answer': example['input'].split("\nLLM response: ")[1].strip()})
            dataset = merge_datasets_on_column_names(dataset_a=dataset,
                                                     dataset_b=reasoning_dataset,
                                                     kept_columns_in_b=['question', 'answer', 'reason'],
                                                     left_on_columns=['question', 'hallucinated_answer'],
                                                     right_on_columns=['question', 'answer'])
            print(f"dataset = {dataset}")
            dataset = dataset.rename_column('reason', 'hallucination_reason')
            dataset = dataset.remove_columns(['answer'])
            print(f"dataset = {dataset}")
            print(f"dataset = {dataset[999]}")
            write_hf_dataset_to_csv(dataset_to_store=dataset, is_append_mode=False,
                                    csv_file_path=self.data_reader.get_file_path('..', 'cache', 'downloaded_data',
                                                                                 dataset_name, 'fixing_training_data',
                                                                                 subset_name + "_fixing.csv"))
        return dataset


if __name__ == '__main__':
    # FixingDataPreparer().process_reasoning_data_for_hallucination("HaluEval-qa")
    FixingDataPreparer().process_reasoning_data_for_hallucination("HaluEval-dialogue")
    # dataset = FixingDataPreparer().process_reasoning_data_for_hallucination("HaluEval-summarization")
