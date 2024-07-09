from experiments.experiment_model_loader import ExperimentModelLoader
from experiments.wrappers.exp_data_processing import url_exp_construct_data
from utils.file_operations import write_a_list_to_file
from wrapper.url_detection_wrapper import URLDetectionWrapper
import time
import sys


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
    write_a_list_to_file(f"wrapper_result-{end_time - start_time}.txt", new_texts)
    return new_texts


def llm_test(llm_name):
    model_loader = ExperimentModelLoader(llm_name)
    pipe = model_loader.load_model()
    dataset = url_exp_construct_data()
    new_texts = []
    start_time = time.time()
    for text in dataset:
        new_texts.append(pipe(model_loader.get_url_detection_prompt(user_query=text)))
    end_time = time.time()
    print(f"Time taken to execute the code: {end_time - start_time} seconds")
    write_a_list_to_file(f"{llm_name}_result-{end_time - start_time}.txt", new_texts)
    return new_texts


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    if MODEL_NAME == "wrapper:":
        wrapper_test()
    else:
        llm_test(MODEL_NAME)
