from transformers import AutoTokenizer
from utils.constants import MODEL_NAME_TINYLAMMA, FOX_INSTRUCT
from datasets import Dataset


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


def merge_datasets_on_column_names(dataset_a, dataset_b, kept_columns_in_b: list,
                                   left_on_columns: list, right_on_columns: list):
    for col_name in left_on_columns:
        dataset_a = dataset_a.map(lambda example: {col_name: example[col_name].strip() if isinstance(example[col_name], str) else example[col_name]})
    for col_name in right_on_columns:
        dataset_b = dataset_b.map(lambda example: {col_name: example[col_name].strip() if isinstance(example[col_name], str) else example[col_name]})
    df_a = dataset_a.to_pandas()
    df_b = dataset_b.to_pandas()
    df_a = df_a.merge(df_b[kept_columns_in_b], left_on=left_on_columns, right_on=right_on_columns, how='left')
    dataset_a = Dataset.from_pandas(df_a)
    return dataset_a