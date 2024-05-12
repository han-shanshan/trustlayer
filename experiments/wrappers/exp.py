from data_operation.data_reader import DataReader
from experiments.model_loader import ModelLoader, TINYLLAMA_MODEL, MISTRAL7B_MODEL, FALCON40B_MODEL, LLAMA3_8B_MODEL, \
    FOX_MODEL
from experiments.wrappers.exp_data_processing import url_exp_construct_data
from wrapper.url_detection_wrapper import URLDetectionWrapper
import time
import sys


def write_to_file(file_name, array):
    with open(file_name, 'w') as f:
        for item in array:
            f.write(f"{item}\n")


def wrapper_test():
    dataset = url_exp_construct_data()
    wrapper = URLDetectionWrapper(config=None)
    new_texts = []
    start_time = time.time()
    for text in dataset:
        new_texts.append(wrapper.process_problematic_urls(text))
    end_time = time.time()
    print(f"Time taken to execute the code: {end_time - start_time} seconds")
    print(new_texts)
    write_to_file(f"wrapper_result-{end_time - start_time}.txt", new_texts)
    return new_texts


def llm_test(llm_name):
    model_loader = ModelLoader(llm_name)
    pipe = model_loader.load_model()
    dataset = url_exp_construct_data()
    new_texts = []
    start_time = time.time()
    for text in dataset:
        new_texts.append(pipe(model_loader.get_url_detection_prompt(user_query=text)))
    end_time = time.time()
    print(f"Time taken to execute the code: {end_time - start_time} seconds")
    write_to_file(f"{llm_name}_result-{end_time - start_time}.txt", new_texts)
    return new_texts


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    if MODEL_NAME == "wrapper:":
        wrapper_test()
    else:
        llm_test(MODEL_NAME)
