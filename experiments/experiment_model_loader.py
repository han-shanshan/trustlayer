from transformers import pipeline
from data_operation.data_file_operator import DataFileOperator
import torch
from utils.constants import FOX

TINYLLAMA_MODEL = "tinyllama"
MISTRAL7B_MODEL = "mistral-7b"
FALCON40B_MODEL = "falcon-40"
LLAMA3_8B_MODEL = "llama3-8b"
LLAMA2_7B_MODEL = "llama2-7b"
FOX_MODEL = "fox"

INFERENCE_PROMPT = "You are a helpful assistant. I will provide some text. Please add a warning before the text " \
                   "without modifying the original " \
                   "content. Do not write any code, but directly return the modified text. Check if the text contains " \
                   "URLs; if it does, list all URLs included in the text. Then, validate whether these URLs are safe, " \
                   "i.e., ensure they are not phishing URLs and are reachable. If any URL is not safe, add a warning " \
                   "before the text and list the unsafe URL. The text is provided below: "


class ExperimentModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        if self.model_name == MISTRAL7B_MODEL:
            hf_key = DataFileOperator().read_hf_apikey()
            print(hf_key)
            return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", return_full_text=False,
                            token=hf_key, max_new_tokens=1024, device=3)
        if self.model_name == TINYLLAMA_MODEL:
            return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", return_full_text=False,
                            device=4)
        if self.model_name == FALCON40B_MODEL:
            return pipeline("text-generation", model="tiiuae/falcon-40b-instruct", return_full_text=False,
                            torch_dtype=torch.bfloat16, max_new_tokens=1024, device_map="auto", trust_remote_code=True)
        if self.model_name == FOX_MODEL:
            return pipeline("text-generation", max_new_tokens=1024, model=FOX, return_full_text=False,
                            device=5)
        if self.model_name == LLAMA3_8B_MODEL:
            hf_key = DataFileOperator().read_hf_apikey()
            return pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_key,
                            return_full_text=False, max_new_tokens=1024, device=6)
        if self.model_name == LLAMA2_7B_MODEL:
            hf_key = DataFileOperator().read_hf_apikey()
            return pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", token=hf_key,
                            return_full_text=False, max_new_tokens=1024, device=7)

    def get_url_detection_prompt(self, user_query):
        """
        reference for llama prompt: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        reference for falcon prompt: https://medium.com/@abirkhan4u/falcon-7b-instruct-llm-with-langchain-empowering-language-processing-like-never-before-b8abb0498bf
        """


        user_instruction = "I will provide some text. Please add a warning before the text without modifying the " \
                           "original content. Do not write any code, but directly return the modified text. Check " \
                           "if the text contains URLs; if it does, list all URLs included in the text. Then, " \
                           "validate whether these URLs are safe, i.e., ensure they are not phishing URLs and are " \
                           "reachable. If any URL is not safe, add a warning before the text and list the unsafe " \
                           "URL. The text is provided below: "
        if self.model_name in [LLAMA3_8B_MODEL, LLAMA2_7B_MODEL]:
            return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> {user_instruction} {user_query}[/INST] "
        elif self.model_name in [FALCON40B_MODEL]:
            return f"You are an intelligent chatbot. Help the following question with brilliant answers. Question: {user_query} Answer: "
        else:
            return user_instruction + user_query
