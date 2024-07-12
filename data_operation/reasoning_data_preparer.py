from datasets import concatenate_datasets, load_dataset
from data_operation.data_file_operator import FileOperator
from utils.file_operations import write_hf_dataset_to_csv
from utils.openai_agent import OpenAIAgent


class ReasoningDataPreparer:
    def __init__(self, agent_type="fedml"):
        self.file_operator = FileOperator()
        self.openai_agent = OpenAIAgent(agent_type=agent_type)  # agent_type="fedml"

    def construct_input_output_pairs_for_hallu_detection(self, question, knowledge, llm_answer, log_type,
                                                         is_hallucination):
        # todo: change qa_openai -- LLM answer->LLM response
        information_input = self.get_input_information(knowledge, llm_answer, log_type, question)
        if is_hallucination:
            detection_result = "Yes"
            reason = self.get_reason_for_hallucination_with_GPT(is_hallucination=is_hallucination, input=information_input)
        else:
            detection_result = "No"
            reason = ""
        return {"input": information_input, "is_hallucination": detection_result, "reason": reason}

    def get_input_information(self, knowledge, llm_answer, log_type, question):
        if log_type not in ["qa", "dialogue", "summarization"]:
            raise ValueError(f"unsupported log type: {log_type}")
        knowledge_template = "Knowledge"
        if log_type == "qa":
            question_template = "Question"
        elif log_type == "dialogue":
            question_template = "Dialogue"
        else:
            question_template = "Question"
            knowledge_template = "Document"
        information_input = f"{question_template}: {question}\n{knowledge_template}: {knowledge}\nLLM response: {llm_answer}"
        return information_input

    @staticmethod
    def _get_GPT_prompt_for_halu_reasoning(is_hallucination, input):
        if is_hallucination:
            question = "Could you explain why there is hallucination in the LLM response?"
        else:
            question = "Could you explain why there is no hallucination in the LLM response? "
        return f"{question} Please use one or two sentences to explain. '{input}' "

    def get_reason_for_hallucination_with_GPT(self, is_hallucination, input):
        prompt = self._get_GPT_prompt_for_halu_reasoning(is_hallucination, input)
        return self.openai_agent.query(prompt=prompt)

    def process_reasoning_data_for_hallucination(self, dataset_type, start_idx=0):
        subset_name = ""
        dataset_name = ""
        if "-" in dataset_type:
            dataset_name = dataset_type.split("-")[0]
            subset_name = dataset_type.split("-")[1]
        dataset = None
        data_chunk_size = 200
        if dataset_type in ["HaluEval-qa", "HaluEval-dialogue", "HaluEval-summarization"]:
            dataset = load_dataset("pminervini/HaluEval", subset_name)["data"]
            # dataset = dataset.select(range(5))  # for testing

            if subset_name == "dialogue":
                dataset = dataset.rename_column('dialogue_history', 'question')
                dataset = dataset.rename_column('right_response', 'right_answer')
                dataset = dataset.rename_column('hallucinated_response', 'hallucinated_answer')
            if subset_name == "summarization":
                dataset = dataset.map(lambda example: {"question": "You are a helpful assistant, please help summarize the document."})
                dataset = dataset.rename_column('right_summary', 'right_answer')
                dataset = dataset.rename_column('hallucinated_summary', 'hallucinated_answer')
                dataset = dataset.rename_column('document', 'knowledge')


            dataset.filter(lambda example: example['question'] is not None and example["knowledge"] is not None
                                           and example["right_answer"] is not None and example[
                                               "hallucinated_answer"] is not None)
            print(dataset)
            for i in range(start_idx, len(dataset), data_chunk_size):
                data_chunk = dataset.select(range(i, min(i + data_chunk_size, len(dataset))))
                data_chunk1 = data_chunk.map(lambda example: self.construct_input_output_pairs_for_hallu_detection(
                    question=example['question'], knowledge=example['knowledge'],
                    llm_answer=example['right_answer'], log_type=subset_name, is_hallucination=False),
                                             remove_columns=dataset.column_names)
                data_chunk2 = data_chunk.map(lambda example: self.construct_input_output_pairs_for_hallu_detection(
                    question=example['question'], knowledge=example['knowledge'],
                    llm_answer=example['hallucinated_answer'], log_type=subset_name, is_hallucination=True),
                                             remove_columns=dataset.column_names)
                data_chunk = concatenate_datasets([data_chunk2, data_chunk1])
                write_hf_dataset_to_csv(dataset_to_store=data_chunk, is_append_mode=True,
                                        csv_file_path=self.file_operator.get_file_path('..', 'cache', 'downloaded_data',
                                                                                       dataset_name,
                                                                                       subset_name + ".csv"))
                # sub_dataset = self.remove_duplicates_in_a_dataset(sub_dataset, col_name1="input", col_name2="output")
                print(f"----------{(i + data_chunk_size)} records done---------")
            data = FileOperator().read_dataset_from_csv_file('..', 'cache', 'downloaded_data', dataset_name,
                                                             subset_name + ".csv")['train'].shuffle(seed=42)
            print(f"data = {data}")
            write_hf_dataset_to_csv(dataset_to_store=data, is_append_mode=False,
                                    csv_file_path=self.file_operator.get_file_path('..', 'cache', 'downloaded_data',
                                                                                   dataset_name,
                                                                                   subset_name + "_shuffled.csv"))
        return dataset