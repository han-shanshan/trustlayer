from openai import OpenAI
from data_operation.data_file_operator import FileOperator


class OpenAIAgent:
    def __init__(self, agent_type="openai"):
        if agent_type == "openai":
            self.client = OpenAI(  # base_url="https://api.openai.com/v1/chat/completions",
                api_key=FileOperator().read_openai_apikey())
            self.model = "gpt-3.5-turbo"
        elif agent_type == "fedml":
            self.client = OpenAI(base_url="https://open.tensoropera.ai/inference/api/v1",
                                 api_key=FileOperator().read_fedml_apikey())
            self.model = "meta/Meta-Llama-3-70B-Instruct"
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}. ")

    def query(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        # print(f"{len(messages[0]['content'])}---message = {messages}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=512
        )
        # print(response)
        return response.choices[0].message.content


if __name__ == '__main__':
    from data_operation.reasoning_data_preparer import ReasoningDataPreparer

    agent = OpenAIAgent(agent_type="fedml")
    q = ReasoningDataPreparer().construct_input_output_pairs_for_hallu_detection(
        question="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.",
        knowledge="Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
        # llm_answer="I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well.", is_hallucination=False,
        llm_answer="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.",
        is_hallucination=True,
        log_type="conversation")
    print(q)

"""
open ai output:
{'input': "Conversation: [Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.\nKnowledge: Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal\nLLM Answer: I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.", 'is hallucination': 'Yes', 'reason': 'The hallucination in the LLM response occurs because the assistant incorrectly states that Robert Downey Jr. starred in Zodiac with Tom Hanks, when in fact he starred in it with Jake Gyllenhaal.'}

llama3 output:
{'input': "Conversation: [Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.\nKnowledge: Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal\nLLM Answer: I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.", 'is hallucination': 'Yes', 'reason': 'The hallucination in the LLM response occurs because the model incorrectly states that Robert Downey Jr. starred in Zodiac with Tom Hanks, when in fact he starred in it with Jake Gyllenhaal.'}

"""
