"""
Company name detection:
https://huggingface.co/Vsevolod/company-names-similarity-sentence-transformer

"""
import pandas as pd
from transformers import pipeline
import re

from storage.vector_db_operator import VectorDBOperator


def read_data_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def retrieve_company_name(text, starting_position):
    substring = text[starting_position:]
    # find the substring ending at the first space or punctuation
    match = re.search(r'^[^ .,!?]*', substring)
    if match:
        return match.group(0)
    else:
        return ""


# todo: remove duplicate keywords in idx, e.g., idx like aaabbb - aaabbb - aaabbb

class HallucinationDataOperator():
    def __init__(self):
        self.brand_extractor = pipeline("token-classification", model="dslim/bert-base-NER")
        # self.pipeline = pipeline("token-classification", model="dslim/bert-base-NER-uncased")
        """
            keyword extraction pipes: summarization_pipe will generate some uncontrolled stuff
            todo: Will be replaced with Fox because the performance is not quite satisfying
        """
        # self.keyword_extractor = pipeline("token-classification", model="yanekyuk/bert-uncased-keyword-extractor")
        # self.summarization_pipe = pipeline("summarization", model="Falconsai/text_summarization")
        self.title_generation_pipe = pipeline("text2text-generation", model="czearing/article-title-generator")
        self.vector_db_operator = VectorDBOperator()
        self.index_name = ""

    @staticmethod
    def read_from_txt_file(file_path, retrieved_col_name):
        # the txt file only contain values in the knowledge column;
        # the file is created by copying the knowledge column from the original data file
        raw_data = read_data_file(file_path)
        data_list = raw_data.split("\"\n\"")
        if retrieved_col_name.lower() == data_list[0].lower():
            data_list = data_list[1:]
        if data_list[0].startswith("\""):
            data_list[0] = data_list[0][1:]
        if data_list[-1].endswith("\""):
            data_list[-1] = data_list[-1][:-1]
        return data_list

    def read_data_from_file(self, file_path, retrieved_col_name="knowledge"):
        file_type = file_path.split(".")[-1].lower()
        if file_type.lower() == "txt":
            return self.read_from_txt_file(file_path=file_path, retrieved_col_name=retrieved_col_name)
        if file_type.lower() == "csv":
            df = pd.read_csv(file_path)
        elif file_type.lower() == "xlsx":
            df = pd.read_excel(file_path)
        else:
            raise TypeError(f"file type {file_type} does not exist. ")
        column_name_exist_flag = False
        for column in df.columns.tolist():
            if retrieved_col_name.lower() in column.lower():
                column_name_exist_flag = True
                if retrieved_col_name != column:
                    retrieved_col_name = column
                break
        if not column_name_exist_flag:
            raise ValueError(f"Column name {retrieved_col_name} does not exist. ")
        return [word for word in list(set(df[retrieved_col_name])) if type(word) is str]

    def create_knowledge_db(self, idx_name):
        raw_knowledge_data = self.read_data_from_file("hallucination_cases.xlsx")
        print(f"len(knowledge) = {len(raw_knowledge_data)}")
        processed_records = self.process_knowledge(raw_knowledge_data, split="---------")
        print(processed_records[0])
        self.index_name = idx_name
        self.vector_db_operator.store_data_to_db(processed_records, idx_name=idx_name)

    def search_in_vector_db(self, text, index=None):
        if index is None:
            index = self.index_name
        self.vector_db_operator.search(text, index)

    def process_knowledge(self, raw_knowledge_data, split="---------"):
        knowledge_dict = {}
        for data in raw_knowledge_data[:1]:
            brand = self.extract_brand_name(data)  # No brand: No-Brand
            qa_pairs = data.split(split)
            for qa in qa_pairs:
                q_and_a = qa.strip().split("Answer:")
                res = self.title_generation_pipe([q_and_a[0].replace("Question:", "").strip(), q_and_a[1].strip()])
                q_keyword = res[0]['generated_text'].strip()
                a_keyword = res[1]['generated_text'].strip()
                if qa not in knowledge_dict:
                    knowledge_dict[qa] = brand + ":" + q_keyword + "|" + a_keyword
                else:  # try to choose a more comprehensive idx between the 2
                    print(f" duplicates !!!!!!")
                    if len(q_keyword + "|" + a_keyword) <= len(
                            knowledge_dict[qa].split(":")[1].strip()):  # the original one is good
                        knowledge_dict[qa] = brand + "," + knowledge_dict[qa]
                    else:  # the new one is good
                        knowledge_dict[qa] = knowledge_dict[qa].split(":")[
                                                 0].strip() + "," + brand + ":" + q_keyword + "|" + a_keyword
        return ["[" + index + "] " + data for data, index in knowledge_dict.items()]

    def extract_brand_name(self, data):
        org_detection_res = self.brand_extractor(data)
        brand_name = "No-Brand"
        for d in org_detection_res:
            if d['entity'] == 'B-ORG':
                brand_name = retrieve_company_name(data, d['start'])
                break
        return brand_name


if __name__ == '__main__':
    p = HallucinationDataOperator()
    p.create_knowledge_db(idx_name="idx.bin")
    p.search_in_vector_db("where is your office?", "idx.bin")
