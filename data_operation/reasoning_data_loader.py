from datasets import DatasetDict
from data_operation.data_loader import DataLoader
from utils.constants import FOX_INSTRUCT, EXPLANATION_RESPONSE_TEMPLATE
from utils.openai_agent import OpenAIAgent


class ReasoningDataLoader(DataLoader):
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer

    def load_reasoning_data(self, dataset_types: list = None, data_num_dict=None, base_model=FOX_INSTRUCT):
        training_dataset, validation_dataset, test_dataset = self.load_hallucination_data_for_reasoning(
            data_num_dict, dataset_types)
        if base_model == FOX_INSTRUCT:
            task_data = self.get_hallu_reasoning_data_for_fox_instruct(training_dataset,
                                                                       validation_dataset,
                                                                       test_dataset)
        else:
            task_data = self.get_hybrid_hallucination_data_for_fox_base(training_dataset,
                                                                        validation_dataset,
                                                                        test_dataset)
        print(f"task data = {task_data}")
        print(f"sample data = {task_data['train'][0]}")
        return task_data

    def get_hallu_reasoning_data_for_fox_instruct(self, training_dataset, validation_dataset, test_dataset):
        print(f"training_dataset = {training_dataset}")
        training_dataset = training_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_instruct(example["input"],
                                                                     example["is_hallucination"],
                                                                     example["reason"])})
        validation_dataset = validation_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_instruct(example["input"],
                                                                     example["is_hallucination"],
                                                                     example["reason"])})
        test_dataset = test_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_instruct(example["input"],
                                                                     example["is_hallucination"],
                                                                     example["reason"])})
        print(f"training_dataset = {training_dataset}")
        datasets = DatasetDict({
            'train': training_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text"]]),
            'validation': validation_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text"]]),
            'test': test_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text"]])
        })
        print(f"final datasets = {datasets}")
        return datasets

    def load_hallucination_data_for_reasoning(self, data_num_dict, dataset_types):
        if data_num_dict is None:
            raise ValueError(f"data num dict is None!")
        print(f"dataset_types = {dataset_types}")
        dataset_list = []
        for dataset_type in dataset_types:
            # print(f"dataset_type = {dataset_type}, type = {type(dataset_type)}")
            if "-" in dataset_type:

                folder_name = dataset_type.split("-")[0]
                data_file_name = dataset_type.split("-")[1] + "_shuffled.csv"
                sub_dataset = self.data_reader.read_dataset_from_csv_file('..', 'cache', 'downloaded_data',
                                                                          folder_name, data_file_name)
            else:
                dataset_dict_from_files = self.data_reader.read_csv_data_files('..', 'cache', 'downloaded_data',
                                                                               dataset_type)
                if len(dataset_dict_from_files) > 0:
                    sub_dataset = self._merge_several_datasets_of_different_phases(dataset_dict_from_files)
                else:
                    raise Exception(f"no data files for dataset {dataset_type}")
            dataset_list.append(sub_dataset['train'])
        # dataset_list = self.remove_duplicates_between_datasets(dataset_list) # todo: when there are more datasets
        training_dataset, validation_dataset, test_dataset = self.create_a_hybrid_dataset_based_on_data_num_dict(
            data_num_dict, dataset_types, dataset_list)
        return training_dataset, validation_dataset, test_dataset

    def get_hybrid_hallucination_data_for_fox_base(self, training_dataset, validation_dataset, test_dataset):
        training_dataset = training_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_base(example["input"], example["output"])})
        validation_dataset = validation_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_base(example["input"], example["output"])})
        test_dataset = test_dataset.map(lambda example: {
            "text": self.get_hallu_reasoning_prompt_for_fox_base(example["input"], "")})

        datasets = DatasetDict({
            'train': training_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text"]]),
            'validation': validation_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text"]]),
            'test': test_dataset.remove_columns(
                [col for col in training_dataset.column_names if col not in ["text", "output"]])
        })
        print(f"final datasets = {datasets}")
        return datasets

    @staticmethod
    def get_hallu_reasoning_prompt_for_fox_base(input_infor, output):
        return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> According to the Question and the Knowledge, " \
               f"is there any hallucination in the LLM Answer?  {input_infor}. {EXPLANATION_RESPONSE_TEMPLATE}[" \
               f"/INST] {output}. "

    def get_hallu_reasoning_prompt_for_fox_instruct(self, input_infor, is_hallucination, reason):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"According to Question/Dialogue and Knowledge, is there any hallucination in the LLM "
                           f"Response? {input_infor}",
            },
            {
                "role": "assistant",
                "content": f"{is_hallucination} {reason}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            # chat_template=template,
            # add_generation_prompt=False for training;
            # add_generation_prompt=True for generation/inference
        )
        return prompt

    # if dataset_type == "rag-hallucination1000":
    #     sub_dataset = load_dataset("neural-bridge/rag-hallucination-dataset-1000")
    #     sub_dataset = self._merge_several_datasets_of_different_phases([sub_dataset])
    #     sub_dataset = sub_dataset.map(lambda example: {"Input": f"Question: {example['question']}; Context: {example['context']}; Answer: {example['answer']}", "Output": "No, the context does not contain necessary information to answer the question. "})
    #     print(sub_dataset)
    #     sub_dataset = self.merge_datasets_of_different_phases_and_remove_duplicates([sub_dataset], col_name1="Input", col_name2="Output")

