import pandas as pd
from transformers import pipeline
import re
from data_operation.data_operator import DataOperator
from data_operation.data_reader import DataReader
from utils.translator import Translator
from data_operation.vector_db_operator import VectorDBOperator


def retrieve_company_name(text, starting_position):
    substring = text[starting_position:]
    # find the substring ending at the first space or punctuation
    match = re.search(r'^[^ .,!?]*', substring)
    if match:
        return match.group(0)
    else:
        return ""


# todo: remove duplicate keywords in idx, e.g., idx like aaabbb - aaabbb - aaabbb
class HallucinationDataOperator(DataOperator):
    """
    brand_extractor pipes: "dslim/bert-base-NER-uncased" performs worse than dslim/bert-base-NER?? todo: to double check with more samples
    keyword extraction pipes:
        summarization pipe "Falconsai/text_summarization" will generate some uncontrolled stuff
        "yanekyuk/bert-uncased-keyword-extractor" will only extract single keywords
    todo: Will be replaced with Fox because the performance is not quite satisfying
    """

    def __init__(self):
        super().__init__()
        self.brand_extractor = pipeline("token-classification", model="dslim/bert-base-NER")
        self.title_generation_pipe = pipeline("text2text-generation", model="czearing/article-title-generator")
        self.vector_db_operator = VectorDBOperator()
        self.index_name = ""

    def create_knowledge_db(self, idx_path, data_file_path="data/hallucination_cases.xlsx"):
        raw_knowledge_data = DataReader.read_data_from_file(data_file_path, retrieved_col_name="knowledge")
        print(f"len(knowledge) = {len(raw_knowledge_data)}")
        idxs, plaintext_knowledge = self.process_knowledge(raw_knowledge_data, split="---------")
        print(f"sample of processed record: idxs[0] = {idxs[0]}")
        print(f"sample of processed record: plaintext_knowledge[0] = {plaintext_knowledge[0]}")
        self.index_name = idx_path
        self.vector_db_operator.store_data_to_vector_db(idxs, idx_name=idx_path)
        df = pd.DataFrame(plaintext_knowledge)
        df.to_csv('plaintext_knowledge_data.csv', index=False)

    def search_in_vector_db(self, text, plaintext_file_path, k=10, index=None):
        if index is None:
            index = self.index_name
        results = self.vector_db_operator.search_vectors(text, index, k)
        df = DataReader.read_data_from_file(plaintext_file_path)

        # join by: df1.ann == data.index
        results = pd.merge(results, df, left_on='ann', right_index=True)
        print(f"retrieved knowledge: \n{results}")
        # results.to_csv('knowledge_data.csv', index=False)
        return results

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
        return [idx for _, idx in knowledge_dict.items()], ["[" + idx + "] " + data for data, idx in
                                                            knowledge_dict.items()]

    def extract_brand_name(self, data):
        _, english_data = Translator().get_instance().language_unification(data)
        org_detection_res = self.brand_extractor(english_data)
        brand_name = "No-Brand"
        for d in org_detection_res:
            if d['entity'] == 'B-ORG':
                brand_name = retrieve_company_name(data, d['start'])
                break
        return brand_name

# if __name__ == '__main__':
#     p = HallucinationDataOperator()
#     p.create_knowledge_db(idx_path="../idx.bin", data_file_path="../data/hallucination_cases.xlsx")
#     p.search_in_vector_db("How to Charge the Camera", k=10, index="idx.bin")
