from hallucination_pipeline.test_use_case import product_knowledge, common_knowledge


class GroundingTool:
    def __init__(self):
        pass

    @staticmethod
    def get_question_topic(text):  # todo: to complete
        # 1. warranty 2. product recommendation 3. return process
        # 4. product information 5. product usage 6. troubleshooting
        question_type = "product_recommendation"
        brand_name = "seagull"  # todo: extract brand name from text
        if "warranty" in text or "waranty" in text:
            # the test data has a typo: warranty --> waranty todo: how to handle typo?
            question_type = "warranty"
        return question_type, brand_name

    def grounding(self, text):  # todo: to complete using vector DB
        question_type, brand_name = self.get_question_topic(text)
        general_knowledge = ""
        general_info_supplementary_qa = ""
        prod_knowledge = ""
        print(f"question_type = {question_type}")
        if question_type == "warranty":
            prod_knowledge, general_knowledge, general_info_supplementary_qa = self.get_warranty_knowledge(brand_name)
        # print(prod_knowledge, general_knowledge, general_info_supplementary_qa)
        return "We received a query from our customer, as follows: " + text + "\n" + str(general_knowledge) + \
               "\n Please find reference information as follows: " + str(prod_knowledge) + \
               "\n if the information is not sufficient, then refer to: " + str(general_info_supplementary_qa)

    def get_warranty_knowledge(self, brand_name):
        general_knowledge, supplementary_qa = self.get_common_knowledge_for_warranty()
        prodknowledge = self.get_product_knowledge_for_warranty(brand_name)
        return prodknowledge, general_knowledge, supplementary_qa

    @staticmethod
    def get_common_knowledge_for_warranty():
        return common_knowledge["warranty"], common_knowledge["waranty_supplymentary_info"]

    @staticmethod
    def get_product_knowledge_for_warranty(brand_name):
        return product_knowledge[brand_name]
