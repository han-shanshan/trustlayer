from transformers import pipeline
from data_operation.data_reader import DataReader
import torch
from multitask_lora.constants import FOX_BASE_GPU

TINYLLAMA_MODEL = "tinyllama"
MISTRAL7B_MODEL = "mistral-7b"
FALCON40B_MODEL = "falcon-40"
LLAMA3_8B_MODEL = "llama3-8b"
FOX_MODEL = "fox"


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
            pipeline("text-generation", model="tiiuae/falcon-40b", torch_dtype=torch.bfloat16,
                     max_new_tokens=1024, device=6)

        if model_name == FOX_MODEL:
            return pipeline("text-generation", model=FOX_BASE_GPU)
        if model_name == LLAMA3_8B_MODEL:
            hf_key = DataReader().read_hf_apikey()
            return pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", token=hf_key)


if __name__ == '__main__':
    ModelLoader.load_model(FALCON40B_MODEL)