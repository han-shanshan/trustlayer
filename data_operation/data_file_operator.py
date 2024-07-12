import pandas as pd
import os
from datasets import load_dataset, DatasetDict


def read_data_from_txt_file(file_path, retrieved_col_name=""):
    # the txt file only contain values in the knowledge column;
    # the file is created by copying the knowledge column from the original data file
    raw_data = FileOperator.read_data_file(file_path)
    data_list = raw_data.split("\"\n\"")
    if len(retrieved_col_name) > 0 and retrieved_col_name.lower() == data_list[0].lower():
        data_list = data_list[1:]
    if data_list[0].startswith("\""):
        data_list[0] = data_list[0][1:]
    if data_list[-1].endswith("\""):
        data_list[-1] = data_list[-1][:-1]
    return data_list


class FileOperator:
    def __init__(self):
        pass

    def read_hf_apikey(self):
        file_path = self.get_file_path('..', 'utils', 'api_keys', 'huggingface.apikey')
        return self.read_data_file(file_path)

    def read_fedml_apikey(self):
        file_path = self.get_file_path('..', 'utils', 'api_keys', 'fedml_api.key')
        return self.read_data_file(file_path)

    def read_azure_apikey(self):
        file_path = self.get_file_path('..', 'utils', 'api_keys', 'azure_api.key')
        return self.read_data_file(file_path)

    def read_openai_apikey(self):
        file_path = self.get_file_path('..', 'utils', 'api_keys', 'openai.key')
        return self.read_data_file(file_path)

    def read_google_apikey(self, is_perspective_api=False):
        if not is_perspective_api:
            file_path = self.get_file_path('..', 'utils', 'api_keys', 'google.apikey')
        else:
            file_path = self.get_file_path('..', 'utils', 'api_keys', 'perspective_api.key')
        return self.read_data_file(file_path)

    @staticmethod
    def read_data_file(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    def read_dataset_from_csv_file(self, *path_components):
        """
        Parameters:
        *path_components (str): Components of the path to the CSV file.

        Returns:
        DataFrame: The dataset read from the CSV file.
        """
        csv_file_path = self.get_file_path(*path_components)
        dataset = load_dataset('csv', data_files=csv_file_path, trust_remote_code=True)
        return dataset

    @staticmethod
    def create_a_folder(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            return False
        return True

    @staticmethod
    def get_file_path(*path_components):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_file_path = os.path.join(dir_path, *path_components)
        print(csv_file_path)
        return csv_file_path

    def read_csv_data_files(self, *path_components):
        folder_path = self.get_file_path(*path_components)
        dataset = DatasetDict()
        if self.create_a_folder(folder_path):
            files = os.listdir(folder_path)
            print(f"files = {files}")
            for f in files:
                print(f"file = {f}")
                if f.endswith('.csv'):
                    csv_file_path = self.get_file_path(folder_path, f)
                    split = f.split(".")[0]
                    sub_dataset = load_dataset('csv', data_files=csv_file_path, trust_remote_code=True)
                    dataset[split] = sub_dataset['train']
        print(f"dataset = {dataset}")
        return dataset

    def check_if_folder_contains_data_files(self, folder_path):
        if self.create_a_folder(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith('.csv'):
                    return True
        return False

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
