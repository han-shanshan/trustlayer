from transformers import pipeline
from data_operation.data_reader import DataReader
import torch

TINYLLAMA_MODEL = "tinyllama"
MISTRAL7B_MODEL = "mistral-7b"
FALCON40B_MODEL = "falcon-40"


class ModelLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_model(model_name):
        if model_name == MISTRAL7B_MODEL:
            hf_key = DataReader().read_hf_apikey()
            print(hf_key)
            return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_key,
                            max_new_tokens=1024, device=7)
        if model_name == TINYLLAMA_MODEL:
            return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=5)
        if model_name == FALCON40B_MODEL:
            return pipeline("text-generation", model="tiiuae/falcon-40b-instruct", torch_dtype=torch.bfloat16,
                            max_new_tokens=1024, device_map="auto")

