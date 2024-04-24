from hallucination_pipeline.data_operator import HallucinationDataOperator
from strengthenllm.translator import Translator


class GroundingTool:
    def __init__(self):
        self.data_operator = HallucinationDataOperator()
        self.index = "idx.bin"

    def grounding(self, text):
        language_type, english_text = Translator().get_instance().language_unification(text)
        query_key = self.data_operator.title_generation_pipe(english_text)[0]['generated_text']
        brand_name = self.data_operator.extract_brand_name(english_text)
        prod_knowledge = self.data_operator.search_in_vector_db(text=brand_name + ":" + query_key, index=self.index)
        query_prompt = "We received a query from our customer, as follows: " + text + "\n" + \
                       "\n Please find reference information as follows: " + str(prod_knowledge)
        if language_type != "en":
            print(f"query_prompt={query_prompt}")
            print(f"language_type = {language_type}")
            query_prompt = query_prompt + " Please answer in the following language: " + language_type
        return query_prompt
