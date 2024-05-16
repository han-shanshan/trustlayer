from datasets import load_dataset, concatenate_datasets
import random
from typing import List, Optional
from data_operation.data_operator import DataOperator
from data_operation.data_reader import DataReader
from transformers import pipeline

from utils.util_functions import write_a_list_to_csv_with_panda

RANDOM_SEED = 0

E_COMMERCE_DATASET = "qgyd2021/e_commerce_customer_service"
AI_MEDICAL_CHAT_DATASET = "ruslanmv/ai-medical-chatbot"
NEWS_SUMMARY_DATASET = "argilla/news-summary"
PATIENT_DOCTOR_CHAT_DATASET = "antareepdey/Patient_doctor_chat"
CHAT_DOCTRO_DATASET = "avaliev/chat_doctor"


def remove_newlines(data_entry):
    for key, value in data_entry.items():
        if isinstance(value, str):  # Check if the field is a string
            data_entry[key] = value.replace("\n", " ")  # Replace newlines with space
    return data_entry


def get_q_in_qa_pair(data_entries):
    for key, value in data_entries.items():
        if isinstance(value, str):  # Check if the field is a string
            data_entries[key] = value.split("###Output:")[0].replace("###Input:", "")
    return data_entries


class VectorDBExpDataOperator(DataOperator):
    def __init__(self):
        super().__init__()
        self.dataset_id = ""
        self.rephrase_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.summarization_pipe = pipeline("summarization", model="Falconsai/text_summarization")

    def rephrase(self, entry: str):
        if self.dataset_id in [E_COMMERCE_DATASET]:
            # the question in the dataset is short; need longer questions, so use tinyllama
            text = self.rephrase_pipe(entry)[0]['generated_text']
            if len(text) > 3 * len(entry) and '. ' in text:
                text = text.split(". ")[0].strip()
            return text
        if self.dataset_id in [CHAT_DOCTRO_DATASET, PATIENT_DOCTOR_CHAT_DATASET]:
            return self.summarization_pipe(entry)[0]['summary_text']

    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id

    def get_columns_to_keep(self):
        columns_to_keep = []
        if self.dataset_id == E_COMMERCE_DATASET:
            columns_to_keep = ["question", "answer"]
        else:
            if self.dataset_id == NEWS_SUMMARY_DATASET:  # 21K
                columns_to_keep = ["text"]
            elif self.dataset_id == PATIENT_DOCTOR_CHAT_DATASET:  # 379K
                columns_to_keep = ["Text"]
            elif self.dataset_id == AI_MEDICAL_CHAT_DATASET:
                columns_to_keep = ["Patient", "Doctor"]
            elif self.dataset_id == CHAT_DOCTRO_DATASET:
                columns_to_keep = ["input", "output"]
        return columns_to_keep

    def get_is_qa(self):
        return self.dataset_id in ["qgyd2021/e_commerce_customer_service", "antareepdey/Patient_doctor_chat",
                                   "ruslanmv/ai-medical-chatbot", "avaliev/chat_doctor"]

    def _load_knowledge_dataset(self, dataset_name):
        if dataset_name == "qgyd2021/e_commerce_customer_service":
            dataset = load_dataset("qgyd2021/e_commerce_customer_service", 'faq')["train"]  # 65 FAQ dataset
        else:
            dataset = super()._load_knowledge_dataset(dataset_name)

        return dataset.map(remove_newlines)

    def create_knowledge_db(self, dataset_id=None, store_path="", knowledge_col: Optional[List[str]] = None,
                            supplementary_info_col: Optional[List[str]] = None, indexing_whole_knowledge=False,
                            indexing_q=True, indexing_a=False, qa_sep: dict = None): # todo: remove suplementary info
        self.dataset_id = dataset_id
        if self.dataset_id == "antareepdey/Patient_doctor_chat":
            qa_sep = {"Q": "###Input:", "A": "###Onput:"}

            supplementary_info_col = "Text"
            dataset_name = dataset_id.split("/")[-1]
            storage_prefix = dataset_name
            if store_path != "":
                storage_prefix = store_path + "/" + dataset_name
            supplementary_info_list = self._load_knowledge_dataset(self.dataset_id)["Text"].map(get_q_in_qa_pair)
            write_a_list_to_csv_with_panda(supplementary_info_list, f'{storage_prefix}_supplementary_data.csv')

        if self.dataset_id == AI_MEDICAL_CHAT_DATASET:
            supplementary_info_col = "Description"
            dataset_name = dataset_id.split("/")[-1]
            storage_prefix = dataset_name
            if store_path != "":
                storage_prefix = store_path + "/" + dataset_name
            supplementary_info_list2 = self._load_knowledge_dataset(self.dataset_id)["Patient"]
            write_a_list_to_csv_with_panda(supplementary_info_list2, f'{storage_prefix}_supplementary2_data.csv')

        if self.dataset_id == "argilla/news-summary":
            supplementary_info_col = "prediction"
            indexing_whole_knowledge = True
        if self.dataset_id == "qgyd2021/e_commerce_customer_service":
            supplementary_info_col = "question"
        if self.dataset_id == CHAT_DOCTRO_DATASET:
            supplementary_info_col = "input"

        super().create_knowledge_db(dataset_id=self.dataset_id, store_path=store_path,
                                    knowledge_col=self.get_columns_to_keep(),
                                    supplementary_info_col=supplementary_info_col,
                                    indexing_whole_knowledge=indexing_whole_knowledge,
                                    indexing_q=indexing_q, indexing_a=indexing_a, qa_sep=qa_sep)

    def _process_knowledge(self, knowledge_dataset, knowledge_col: Optional[List[str]] = None,
                           supplementary_info_col: str = None, indexing_whole_knowledge=False,
                           indexing_q=True, indexing_a=False, qa_sep: dict = None):
        plaintext_idxs, plaintext_knowledge, supplementary_info = super()._process_knowledge(knowledge_dataset,
                                                                                             knowledge_col=knowledge_col,
                                                                                             supplementary_info_col=supplementary_info_col,
                                                                                             indexing_whole_knowledge=indexing_whole_knowledge,
                                                                                             indexing_q=indexing_q,
                                                                                             indexing_a=indexing_a,
                                                                                             qa_sep=qa_sep)
        return plaintext_idxs, plaintext_knowledge, supplementary_info


def exp_searching(dataset_id, total_query_num=50, store_path="", is_rephrasing_query=True):
    operator = VectorDBExpDataOperator()
    operator.set_dataset_id(dataset_id)
    storage_prefix = dataset_id.split('/')[-1]
    if store_path != "":
        storage_prefix = store_path + "/" + storage_prefix
    idx_name = storage_prefix + "_idx.bin"
    # full_idx_name = "full_" + dataset_name + "_idx.bin"
    plaintext_knowledge_file_name = storage_prefix + "_knowledge_data.csv"
    question_file_name = f"{storage_prefix}_supplementary_data.csv"
    if dataset_id == AI_MEDICAL_CHAT_DATASET and not is_rephrasing_query:  # extract queries from knowledge file
        question_file_name = f"{storage_prefix}_supplementary2_data.csv"

    df = DataReader.read_data_from_file(question_file_name)
    col_name = df.columns[0]
    original_queries = df.sample(n=total_query_num, random_state=RANDOM_SEED)
    # print(f"queries = {queries}")
    query_index_ids = original_queries.index.to_list()
    # print(f"query_index_ids = {query_index_ids}")
    top1_call_back_counter = 0
    top3_call_back_counter = 0
    top5_call_back_counter = 0
    top10_call_back_counter = 0

    queries = []
    if is_rephrasing_query and dataset_id != AI_MEDICAL_CHAT_DATASET:
        for i in range(total_query_num):
            old_query = original_queries.iloc[i][col_name]
            query = operator.rephrase(old_query)
            print(f"old query = {old_query}, new query = {query}")
            queries.append(query)
    else:
        queries = [original_queries.iloc[i][col_name] for i in range(len(original_queries))]

    for i in range(total_query_num):
        print(f"queries[i] = {queries[i]}")
        result_df = operator.search_in_vector_db_with_index(queries[i], plaintext_knowledge_file_name, k=10,
                                                            index=idx_name)
        ann = result_df["ann"].tolist()
        if query_index_ids[i] in ann:
            top10_call_back_counter += 1
        if query_index_ids[i] in ann[:5]:
            top5_call_back_counter += 1
        if query_index_ids[i] in ann[:3]:
            top3_call_back_counter += 1
        if query_index_ids[i] == ann[0]:
            top1_call_back_counter += 1
        # print(f"retrieved knowledge: \n{result_df}")
    print(f"total_query_num = {total_query_num}")
    print(f"call back top1: call back queries: {top1_call_back_counter}, {top1_call_back_counter / total_query_num}")
    print(f"call back top3: call back queries: {top3_call_back_counter}, {top3_call_back_counter / total_query_num}")
    print(f"call back top5: call back queries: {top5_call_back_counter},{top5_call_back_counter / total_query_num}")
    print(f"call back top10: call back queries: {top10_call_back_counter}, {top10_call_back_counter / total_query_num}")


def exp_indexing_whole_message_original_queries(dataset_id, total_query_num):
    """
    call back top1: call back queries: 50, 1.0
call back top3: call back queries: 50, 1.0
call back top5: call back queries: 50,1.0
call back top10: call back queries: 50, 1.0
    """
    store_path = "exp_indexing_whole_message_original_queries"
    operator = VectorDBExpDataOperator()
    operator.create_knowledge_db(dataset_id=dataset_id, store_path=store_path,
                                 indexing_whole_knowledge=True)
    exp_searching(dataset_id, store_path=store_path, total_query_num=total_query_num, is_rephrasing_query=False)


def exp_indexing_whole_message_rephrased_queries(dataset_id, total_query_num):
    """
    call back top1: call back queries: 33, 0.66
call back top3: call back queries: 39, 0.78
call back top5: call back queries: 40,0.8
call back top10: call back queries: 41, 0.82
    """
    store_path = "exp_indexing_whole_message_rephrased_queries"
    operator = VectorDBExpDataOperator()
    operator.create_knowledge_db(dataset_id=dataset_id, store_path=store_path,
                                 indexing_whole_knowledge=True)
    exp_searching(dataset_id, store_path=store_path, total_query_num=total_query_num, is_rephrasing_query=True)


def exp_indexing_q_original_queries(dataset_id, total_query_num):
    """
    call back top1: call back queries: 50, 1.0
    call back top3: call back queries: 50, 1.0
    call back top5: call back queries: 50,1.0
    call back top10: call back queries: 50, 1.0
    """
    store_path = "exp_indexing_q_original_queries"
    operator = VectorDBExpDataOperator()
    operator.create_knowledge_db(dataset_id=dataset_id, store_path=store_path,
                                 indexing_whole_knowledge=False)
    exp_searching(dataset_id, store_path=store_path, total_query_num=total_query_num, is_rephrasing_query=False)


def exp_indexing_q_rephrased_queries(dataset_id, total_query_num):
    """
    CHAT_DOCTRO_DATASET
    call back top1: call back queries: 44, 0.88
    call back top3: call back queries: 47, 0.94
    call back top5: call back queries: 48,0.96
    call back top10: call back queries: 49, 0.98
    """
    store_path = "exp_indexing_q_rephrased_queries"
    operator = VectorDBExpDataOperator()
    operator.create_knowledge_db(dataset_id=dataset_id, store_path=store_path,
                                 indexing_whole_knowledge=False)
    exp_searching(dataset_id, store_path=store_path, total_query_num=total_query_num, is_rephrasing_query=True)


if __name__ == '__main__':
    """
    https://huggingface.co/datasets/argilla/news-summary?row=0
    medical with concise questions: https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot
    https://huggingface.co/datasets/antareepdey/Patient_doctor_chat?row=0
    https://huggingface.co/datasets/avaliev/chat_doctor?row=0
    e-commercial dataset: https://huggingface.co/datasets/qgyd2021/e_commerce_customer_service?row=33
    """
    # exp_indexing_q_rephrased_queries(CHAT_DOCTRO_DATASET, total_query_num=50)
    # exp_indexing_q_original_queries(CHAT_DOCTRO_DATASET, total_query_num=50)
    # exp_indexing_whole_message_rephrased_queries(CHAT_DOCTRO_DATASET, total_query_num=50)
    # exp_indexing_whole_message_original_queries(CHAT_DOCTRO_DATASET, total_query_num=50)
    # exp_indexing_q_rephrased_queries(AI_MEDICAL_CHAT_DATASET, total_query_num=50)
    # exp_indexing_q_original_queries(AI_MEDICAL_CHAT_DATASET, total_query_num=50)
    # exp_indexing_whole_message_rephrased_queries(AI_MEDICAL_CHAT_DATASET, total_query_num=50)
    exp_indexing_whole_message_original_queries(AI_MEDICAL_CHAT_DATASET, total_query_num=50)
