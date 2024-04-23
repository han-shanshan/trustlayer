import pandas as pd
from transformers import pipeline
import re
from strengthenllm.translator import Translator
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
    """
    brand_extractor pipes: "dslim/bert-base-NER-uncased" performs worse than dslim/bert-base-NER?? todo: to double check with more samples
    keyword extraction pipes:
        summarization pipe "Falconsai/text_summarization" will generate some uncontrolled stuff
        "yanekyuk/bert-uncased-keyword-extractor" will only extract single keywords
    todo: Will be replaced with Fox because the performance is not quite satisfying
    """

    def __init__(self):
        self.brand_extractor = pipeline("token-classification", model="dslim/bert-base-NER")
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

    def create_knowledge_db(self, idx_path):
        raw_knowledge_data = self.read_data_from_file("data/hallucination_cases.xlsx")
        print(f"len(knowledge) = {len(raw_knowledge_data)}")
        processed_records = self.process_knowledge(raw_knowledge_data, split="---------")
        print(processed_records[0])
        self.index_name = idx_path
        self.vector_db_operator.store_data_to_db(processed_records, idx_name=idx_path)

    def search_in_vector_db(self, text, k=10, index=None):
        if index is None:
            index = self.index_name
        return self.vector_db_operator.search(text, index, k)

    def _extract_idx_for_a_qa(self, qa, brand, existing_knowledge=None):
        q_and_a = qa.strip().split("Answer:")
        if len(q_and_a) <= 1:
            return None
        _, english_q = Translator().get_instance().language_unification(q_and_a[0].replace("Question:", "").strip())
        _, english_a = Translator().get_instance().language_unification(q_and_a[1].strip())
        # print(f"english_q = {english_q}, english_a = {english_a}")
        res = self.title_generation_pipe([english_q, english_a])
        # print(f"res = {res}")
        q_keyword = res[0]['generated_text'].strip()
        a_keyword = res[1]['generated_text'].strip()
        # print(f"q = {q_keyword} + a = {a_keyword}")
        if existing_knowledge is None:
            return brand + ":" + q_keyword + "|" + a_keyword
        else:  # try to choose a more comprehensive idx between the 2
            print(f" duplicates !!!!!! original qa idx = {existing_knowledge}")
            if len(q_keyword + "|" + a_keyword) <= len(
                    existing_knowledge.split(":")[1].strip()) and brand not in existing_knowledge.split(":")[
                0].split(","):  # the original one is good
                return brand + "," + existing_knowledge
            else:  # the new one is good
                if brand not in existing_knowledge.split(":")[0].split(","):
                    return existing_knowledge.split(":")[0].strip() + "," + brand + ":" + q_keyword + "|" + a_keyword
                else:
                    return existing_knowledge.split(":")[0].strip() + ":" + q_keyword + "|" + a_keyword

    def process_knowledge(self, raw_knowledge_data, split="---------"):
        knowledge_dict = {}
        for data in raw_knowledge_data:
            brand = self.extract_brand_name(data)  # No brand: No-Brand
            qa_pairs = data.split(split)
            qa_pairs = [item.strip() for item in qa_pairs if item.strip()]
            for qa in qa_pairs:
                existing_knowledge = None
                if qa in knowledge_dict:
                    existing_knowledge = knowledge_dict[qa]
                idx = self._extract_idx_for_a_qa(qa, brand, existing_knowledge)
                if idx is not None:
                    knowledge_dict[qa] = idx
                print(f"idx = {idx}")
        return ["[" + idx + "] " + data for data, idx in knowledge_dict.items()]

    def extract_brand_name(self, data):
        _, english_data = Translator().get_instance().language_unification(data)
        org_detection_res = self.brand_extractor(english_data)
        brand_name = "No-Brand"
        for d in org_detection_res:
            if d['entity'] == 'B-ORG':
                brand_name = retrieve_company_name(data, d['start'])
                break
        return brand_name


if __name__ == '__main__':
    p = HallucinationDataOperator()
    p.create_knowledge_db(idx_path="idx.bin")
    p.search_in_vector_db("How to Charge the Camera", k=10, index="idx.bin")
