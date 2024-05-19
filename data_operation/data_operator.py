import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
import re
from data_operation.data_reader import DataReader
from data_operation.vector_db_operator import VectorDBOperator
from typing import List, Optional
from utils.file_operations import write_a_dictionary_to_file, write_a_list_to_csv_with_panda, load_a_dictionary_from_file


def retrieve_company_name(text, starting_position):
    substring = text[starting_position:]
    # find the substring ending at the first space or punctuation
    match = re.search(r'^[^ .,!?]*', substring)
    if match:
        return match.group(0)
    else:
        return ""


# todo: remove duplicate keywords in idx, e.g., idx like aaabbb - aaabbb - aaabbb
class DataOperator:
    """
    brand_extractor pipes: "dslim/bert-base-NER-uncased" performs worse than dslim/bert-base-NER?? todo: to double check with more samples
    keyword extraction pipes:
        summarization pipe "Falconsai/text_summarization" will generate some uncontrolled stuff
        "yanekyuk/bert-uncased-keyword-extractor" will only extract single keywords
    todo: Will be replaced with Fox because the performance is not quite satisfying
    """
    def __init__(self):
        self.title_generation_pipe = pipeline("text2text-generation", model="czearing/article-title-generator")
        self.vector_db_operator = VectorDBOperator()
        self.dataset_name = ""

    def _generate_summarization_for_an_entry(self, entry: str):
        return self.title_generation_pipe(entry)[0]['generated_text'].strip()

    # def _generate_summarizations_for_a_list(self, entry_list: list):
    #     res_list = self.title_generation_pipe(entry_list)
    #     return [res_list[i]['generated_text'].strip() for i in range(len(res_list))]

    def _generate_summarizations_for_a_list(self, entry_list: list):
        return entry_list

    def _load_knowledge_dataset(self, dataset_name):
        """
        load the dataset and merge subsets (if any)
        example of returned dataset structure:
            Dataset({
                features: ['text', 'prediction'],
                num_rows: 21417
            })
        """
        original_dataset = load_dataset(dataset_name)
        dataset = None
        for split in original_dataset.keys():
            if dataset is None:
                dataset = original_dataset[split]
            else:
                dataset = concatenate_datasets([dataset, original_dataset[split]])
        return dataset

    def create_knowledge_db(self, dataset_id=None, store_path="", knowledge_col: Optional[List[str]] = None,
                            supplementary_info_col: str = None, indexing_whole_knowledge=False,
                            indexing_q=True, indexing_a=False, qa_sep=None):
        raw_knowledge_data = self._load_knowledge_dataset(dataset_id)
        print(f"knowledge_dataset={raw_knowledge_data}, "
              f"knowledge_col={knowledge_col}, "
              f"supplementary_info_col={supplementary_info_col}, indexing_whole_knowledge={indexing_whole_knowledge}, "
              f"indexing_q = {indexing_q}, indexing_a = {indexing_a}, qa_sep={qa_sep}")
        plaintext_idxs, plaintext_knowledge, supplementary_info = self._process_knowledge(
            knowledge_dataset=raw_knowledge_data, knowledge_col=knowledge_col,
            supplementary_info_col=supplementary_info_col, indexing_whole_knowledge=indexing_whole_knowledge,
            indexing_q=indexing_q, indexing_a=indexing_a, qa_sep=qa_sep)

        print(f"plaintext_idxs[0] = {plaintext_idxs[0]}")
        print(f"sample of processed record: plaintext_knowledge[0] = {plaintext_knowledge[0]}")
        dataset_name = dataset_id.split("/")[-1]
        storage_prefix = dataset_name
        if store_path != "":
            storage_prefix = store_path + "/" + dataset_name
        meta_info = {"index_path": f"{storage_prefix}_idx.bin",
                     "knowledge_path": f'{storage_prefix}_knowledge_data.csv',
                     "supplementary_storage_path": f'{storage_prefix}_supplementary_data.csv'}
        self.vector_db_operator.store_data_to_vector_db(plaintext_idxs, idx_name=meta_info['index_path'])
        write_a_list_to_csv_with_panda(plaintext_knowledge, meta_info['knowledge_path'])
        if len(supplementary_info) > 0:
            write_a_list_to_csv_with_panda(supplementary_info, meta_info['supplementary_storage_path'])
        write_a_dictionary_to_file(file_name=dataset_name, dictionary=meta_info)

    """
    knowledge_dataset: loaded dataset
    knowledge_col: Optional[List[str]] = None: informative columns in the dataset that will be used as knowledge 
    is_qa=True: is the dataset a QA dataset or not
    qa_sep=None: separators for QA entries. Usually, a qa entry is in this format: "Question: xxx. Answer: xxx"
    """

    def _process_knowledge(self, knowledge_dataset, knowledge_col: Optional[List[str]] = None,
                           supplementary_info_col: str = None, indexing_whole_knowledge=False,
                           indexing_q=True, indexing_a=False, qa_sep: dict = None):
        if knowledge_col is None:
            knowledge_col = knowledge_dataset.column_names
        knowledge_list = []
        plaintext_index_list = []
        supplementary_info_list = []

        if indexing_whole_knowledge:  # merge all columns to create one value
            for data in knowledge_dataset:
                knowledge_entry = ""
                for col in knowledge_col:
                    knowledge_entry = knowledge_entry + data[col] + "|"
                # plaintext_index_list.append(self._generate_summarization_for_an_entry(knowledge_entry))
                if qa_sep is not None:
                    knowledge_entry = knowledge_entry.replace(qa_sep["Q"], "").replace(qa_sep["A"], "")
                plaintext_index_list.append(knowledge_entry)
                knowledge_list.append(knowledge_entry)
        elif len(knowledge_col) == 1:  # may need to split data
            index_method = None
            if indexing_q and not indexing_a:
                index_method = self._extract_q_idx_for_a_qa_string
            elif indexing_a and not indexing_q:
                index_method = self._extract_a_idx_for_a_qa_string
            elif indexing_q and indexing_a:
                index_method = self._extract_qa_idx_for_a_qa_string
            for data in knowledge_dataset:
                qa = data[knowledge_col[0]]
                plaintext_index_list.append(index_method(qa, qa_identifiers=qa_sep))
                knowledge_list.append(data[knowledge_col[0]].replace(qa_sep["Q"], "").replace(qa_sep["A"], ""))
        elif len(knowledge_col) == 2:
            index_method = None
            if indexing_q and not indexing_a:
                index_method = self._extract_q_idx_for_a_list
            elif indexing_a and not indexing_q:
                index_method = self._extract_a_idx_for_a_list
            elif indexing_q and indexing_a:
                index_method = self._extract_combined_idx_for_a_list
            for data in knowledge_dataset:
                knowledge_entry = data[knowledge_col[0]].strip() + " | " + data[knowledge_col[1].strip()]
                plaintext_index_list.append(
                    index_method([data[knowledge_col[0]].strip(), data[knowledge_col[1].strip()]]))
                knowledge_list.append(knowledge_entry)
        else:
            raise Exception("Error")

        if supplementary_info_col is not None:
            supplementary_info_list = knowledge_dataset[supplementary_info_col]

        return plaintext_index_list, knowledge_list, supplementary_info_list

    ############### when inputs are qa strings #####################
    def _get_separate_q_and_a_summarizations(self, qa, qa_identifiers=None):
        if qa_identifiers is None:
            qa_identifiers = {"Q": "Answer:", "A": "Question:"}
        q_and_a = qa.strip().split(qa_identifiers["A"])
        if len(q_and_a) <= 1:
            return None
        question = q_and_a[0].replace(qa_identifiers["Q"], "")
        answer = q_and_a[1].strip()
        # print(f"q = {question}")
        # print(f"a = {answer}")
        return self._generate_summarizations_for_a_list([question, answer])

    def _extract_q_idx_for_a_qa_string(self, qa, qa_identifiers=None):
        return self._get_separate_q_and_a_summarizations(qa, qa_identifiers=qa_identifiers)[0]

    def _extract_a_idx_for_a_qa_string(self, qa, qa_identifiers=None):
        return self._get_separate_q_and_a_summarizations(qa, qa_identifiers=qa_identifiers)[1]

    def _extract_qa_idx_for_a_qa_string(self, qa, qa_identifiers=None):
        q_a_summarizations = self._get_separate_q_and_a_summarizations(qa, qa_identifiers=qa_identifiers)[0]
        return q_a_summarizations[0] + "|" + q_a_summarizations[1]

    ####################################################################

    ##################when inputs are separate Q and A ############################
    def _extract_q_idx_for_a_list(self, entries: list):
        return self._generate_summarizations_for_a_list(entries)[0]

    def _extract_a_idx_for_a_list(self, entries: list):
        return self._generate_summarizations_for_a_list(entries)[1]

    def _extract_combined_idx_for_a_list(self, entries: list):
        summarizations = self._generate_summarizations_for_a_list(entries)
        res = ""
        for i in range(len(summarizations) - 1):
            res = res + summarizations[i] + "|"
        return res + summarizations[-1]

    ##############################################################################

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def search_in_vector_db(self, text, dataset_name, k):
        meta_info = load_a_dictionary_from_file(dataset_name)
        return self.search_in_vector_db_with_index(text=text, plaintext_file_path=meta_info['knowledge_path'],
                                                   k=k, index=meta_info['index_path'])

    def search_in_vector_db_with_index(self, text, plaintext_file_path, k=10, index=None):
        if index is None:
            raise Exception(f"index is {index}")
        results = self.vector_db_operator.search_vectors(text, index, k)
        df = DataReader.read_data_from_file(plaintext_file_path)
        # join by: df1.ann == data.index
        results = pd.merge(results, df, left_on='ann', right_index=True)
        return results

# if __name__ == '__main__':
#     p = HallucinationDataOperator()
#     p.create_knowledge_db(idx_path="../idx.bin", data_file_path="../data/hallucination_cases.xlsx")
#     p.search_in_vector_db("How to Charge the Camera", k=10, index="idx.bin")
