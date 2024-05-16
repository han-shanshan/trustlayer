import json
import pandas as pd


def write_a_list_to_file(file_name, array: list):
    with open(file_name, 'w') as f:
        for item in array:
            f.write(f"{item}\n")


def write_a_dictionary_to_file(file_name, dictionary: dict):
    with open(file_name, 'w') as file:
        file.write(json.dumps(dictionary))


def load_a_dictionary_from_file(file_name):
    with open(file_name) as f:
        data = f.read()
    return json.loads(data)


def write_a_list_to_csv_with_panda(data, file_path):
    if type(data) is list:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    else:
        data.to_csv(file_path, index=False)
