import pandas as pd
import os
from datasets import load_dataset


def read_data_from_txt_file(file_path, retrieved_col_name=""):
    # the txt file only contain values in the knowledge column;
    # the file is created by copying the knowledge column from the original data file
    raw_data = DataReader.read_data_file(file_path)
    data_list = raw_data.split("\"\n\"")
    if len(retrieved_col_name) > 0 and retrieved_col_name.lower() == data_list[0].lower():
        data_list = data_list[1:]
    if data_list[0].startswith("\""):
        data_list[0] = data_list[0][1:]
    if data_list[-1].endswith("\""):
        data_list[-1] = data_list[-1][:-1]
    return data_list


class DataReader:
    def __init__(self):
        pass

    def read_hf_apikey(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, '..', 'utils', 'huggingface.apikey')
        return self.read_data_file(file_path)

    def read_google_apikey(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, '..', 'utils', 'google.apikey')
        return self.read_data_file(file_path)

    @staticmethod
    def read_data_file(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    @staticmethod
    def read_csv_file_data(csv_file_path):
        return load_dataset('csv', data_files=csv_file_path)

    @staticmethod
    def read_data_from_file(file_path, retrieved_col_name="all"):  # "all": retrieve all column
        file_type = file_path.split(".")[-1].lower()
        if file_type.lower() == "txt":
            return read_data_from_txt_file(file_path=file_path, retrieved_col_name=retrieved_col_name)
        if file_type.lower() == "csv":
            df = pd.read_csv(file_path)
        elif file_type.lower() == "xlsx":
            df = pd.read_excel(file_path)
        else:
            raise TypeError(f"file type {file_type} does not exist. ")
        if retrieved_col_name == "all":
            return df
        column_name_exist_flag = False
        for column in df.columns.tolist():
            if retrieved_col_name.lower() in column.lower():
                column_name_exist_flag = True
                if retrieved_col_name != column:
                    retrieved_col_name = column
                break
        if not column_name_exist_flag:
            raise ValueError(f"Column name {retrieved_col_name} does not exist. ")
        return [word for word in list(set(df[retrieved_col_name])) if type(word) is str]
