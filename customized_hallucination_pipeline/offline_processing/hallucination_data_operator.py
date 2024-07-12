from transformers import pipeline
import re
from data_operation.data_operator import DataOperator
from data_operation.data_file_operator import FileOperator
from utils.translator import Translator
from typing import List, Optional


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

    def _load_knowledge_dataset(self, dataset_name=None):
        data_file_path = "data/hallucination_cases.xlsx"
        return FileOperator.read_data_from_file(data_file_path, retrieved_col_name="knowledge")

    def create_knowledge_db(self, dataset_id=None, store_path="", knowledge_col: Optional[List[str]] = None,
                            supplementary_info_col: str = None, indexing_whole_knowledge=False,
                            indexing_q=True, indexing_a=False, qa_sep: dict = None):
        super().create_knowledge_db(dataset_id="hallucination_knowledge", store_path=store_path)

    def search_in_vector_db_with_index(self, text, plaintext_file_path, k=10, index=None):
        return super().search_in_vector_db_with_index(text, plaintext_file_path, k=k, index=index)

    def _process_knowledge(self, knowledge_dataset, knowledge_col: Optional[List[str]] = None,
                           supplementary_info_co=None,
                           indexing_whole_knowledge=False, indexing_q=True, indexing_a=False, qa_sep: dict = None):
        split = "---------"
        knowledge_dict = {}
        for data in knowledge_dataset:
            brand = self.extract_brand_name(data)  # No brand: No-Brand
            qa_pairs = data.split(split)
            qa_pairs = [item.strip() for item in qa_pairs if item.strip()]
            for qa in qa_pairs:
                existing_knowledge = None
                if qa in knowledge_dict:
                    existing_knowledge = knowledge_dict[qa]
                idx = self._extract_idx_for_a_qa_for_customer_knowledge(qa, brand,
                                                                        existing_knowledge=existing_knowledge)
                if idx is not None:
                    knowledge_dict[qa] = idx
                print(f"idx = {idx}")
        return [idx for _, idx in knowledge_dict.items()], ["[" + idx + "] " + data for data, idx in
                                                            knowledge_dict.items()], []

    def _extract_idx_for_a_qa_for_customer_knowledge(self, qa, brand, existing_knowledge=None, qa_identifiers=None):
        if qa_identifiers is None:
            qa_identifiers = {"Q": "Answer:", "A": "Question:"}
        q_and_a = qa.strip().split(qa_identifiers["A"])
        if len(q_and_a) <= 1:
            return None
        _, english_q = Translator().get_instance().language_unification(
            q_and_a[0].replace(qa_identifiers["Q"], "").strip())
        _, english_a = Translator().get_instance().language_unification(q_and_a[1].strip())
        res = self.title_generation_pipe([english_q, english_a])
        q_keyword = res[0]['generated_text'].strip()
        a_keyword = res[1]['generated_text'].strip()
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
#     p.create_knowledge_db(store_path="../idx.bin", data_file_path="../data/hallucination_cases.xlsx")
#     p.search_in_vector_db("How to Charge the Camera", k=10, index="idx.bin")
