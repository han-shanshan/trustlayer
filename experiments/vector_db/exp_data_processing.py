from datasets import load_dataset, concatenate_datasets
import random
from typing import List, Optional

from data_operation.data_operator import DataOperator
from data_operation.data_reader import DataReader

RANDOM_SEED = 0


def remove_newlines(data_entry):
    for key, value in data_entry.items():
        if isinstance(value, str):  # Check if the field is a string
            data_entry[key] = value.replace("\n", " ")  # Replace newlines with space
    return data_entry


class VectorDBExpDataOperator(DataOperator):
    def __init__(self):
        super().__init__()
        self.dataset_id = ""

    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id

    def get_columns_to_keep(self):
        columns_to_keep = []
        if self.dataset_id == "qgyd2021/e_commerce_customer_service":
            columns_to_keep = ["question", "answer"]
        else:
            if self.dataset_id == "argilla/news-summary":  # 21K
                columns_to_keep = ["text"]
            elif self.dataset_id == "antareepdey/Patient_doctor_chat":  # 379K
                columns_to_keep = ["Text"]
            elif self.dataset_id == "ruslanmv/ai-medical-chatbot":
                columns_to_keep = ["Patient", "Doctor"]
            elif self.dataset_id == "avaliev/chat_doctor":
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
                            supplementary_info_col: str = None, is_qa=True, qa_sep=None):
        self.dataset_id = dataset_id
        if self.dataset_id == "antareepdey/Patient_doctor_chat":
            qa_sep = {"Q": "###Input:", "A": "###Onput:"}
        if self.dataset_id == "ruslanmv/ai-medical-chatbot":
            supplementary_info_col = "Description"
        if self.dataset_id == "argilla/news-summary":
            supplementary_info_col = "prediction"
        if self.dataset_id == "qgyd2021/e_commerce_customer_service":
            supplementary_info_col = "question"
        super().create_knowledge_db(dataset_id=self.dataset_id, store_path=store_path,
                                    knowledge_col=self.get_columns_to_keep(),
                                    supplementary_info_col=supplementary_info_col, is_qa=self.get_is_qa(),
                                    qa_sep=qa_sep)

    def _process_knowledge(self, knowledge_dataset, knowledge_col: Optional[List[str]] = None,
                           supplementary_info_col: str = None, is_qa=True, qa_sep=None):
        plaintext_idxs, plaintext_knowledge, supplementary_info = super()._process_knowledge(knowledge_dataset,
                                                                                             knowledge_col=knowledge_col,
                                                                                             supplementary_info_col=supplementary_info_col,
                                                                                             is_qa=is_qa, qa_sep=qa_sep)
        self.vector_db_operator.store_data_to_vector_db(plaintext_knowledge,
                                                        idx_name=f"full_{self.dataset_id.split('/')[-1]}_idx.bin")
        return plaintext_idxs, plaintext_knowledge, supplementary_info


def index_text(dataset_id, total_query_num=50):
    operator = VectorDBExpDataOperator()
    operator.set_dataset_id(dataset_id)
    dataset_name = dataset_id.split('/')[-1]
    idx_name = dataset_name + "_idx.bin"
    # full_idx_name = "full_" + dataset_name + "_idx.bin"
    plaintext_knowledge_file_name = dataset_name + "_knowledge_data.csv"
    question_file_name = f"{dataset_name}_supplementary_data.csv"

    df = DataReader.read_data_from_file(question_file_name)
    col_name = df.columns[0]
    queries = df.sample(n=total_query_num, random_state=RANDOM_SEED)
    # print(f"queries = {queries}")
    query_index_ids = queries.index.to_list()
    # print(f"query_index_ids = {query_index_ids}")

    top3_call_back_counter = 0
    top5_call_back_counter = 0
    top10_call_back_counter = 0
    for i in range(total_query_num):
        query = queries.iloc[i][col_name]
        # print(f"query = {query}")
        result_df = operator.search_in_vector_db_with_index(query, plaintext_knowledge_file_name, k=10, index=idx_name)
        ann = result_df["ann"].tolist()
        # print(f"ann = {ann}")
        if query_index_ids[i] in ann:
            top10_call_back_counter += 1
        if query_index_ids[i] in ann[:5]:
            top5_call_back_counter += 1
        if query_index_ids[i] in ann[:3]:
            top3_call_back_counter += 1
        # print(f"retrieved knowledge: \n{result_df}")
    print(f"total_query_num = {total_query_num}")
    print(f"call back top3: call back queries: {top3_call_back_counter}, {top3_call_back_counter / total_query_num}")
    print(f"call back top5: call back queries: {top5_call_back_counter},{top5_call_back_counter / total_query_num}")
    print(f"call back top10: call back queries: {top10_call_back_counter}, {top10_call_back_counter / total_query_num}")


E_COMMERCE_DATASET = "qgyd2021/e_commerce_customer_service"
AI_MEDICAL_CHAT_DATASET = "ruslanmv/ai-medical-chatbot"
NEWS_SUMMARY_DATASET = "argilla/news-summary"
PATIENT_DOCTOR_CHAT_DATASET = "antareepdey/Patient_doctor_chat"
CHAT_DOCTRO_DATASET = "avaliev/chat_doctor"

if __name__ == '__main__':
    """
    https://huggingface.co/datasets/argilla/news-summary?row=0
    medical with concise questions: https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot
    https://huggingface.co/datasets/antareepdey/Patient_doctor_chat?row=0
    https://huggingface.co/datasets/avaliev/chat_doctor?row=0
    e-commercial dataset: https://huggingface.co/datasets/qgyd2021/e_commerce_customer_service?row=33
    """

    operator = VectorDBExpDataOperator()
    operator.create_knowledge_db(dataset_id=E_COMMERCE_DATASET)
    # index_text(E_COMMERCE_DATASET, total_query_num=50)
