from transformers import AutoTokenizer
from utils.constants import MODEL_NAME_TINYLAMMA, FOX_INSTRUCT


def get_tokenizer(base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if base_model_name in [MODEL_NAME_TINYLAMMA, FOX_INSTRUCT]:
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = 'right'  # to prevent warnings
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id
    else:
        raise ValueError(f"Unsupported model: {base_model_name}")
    return tokenizer