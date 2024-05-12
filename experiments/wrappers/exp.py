from data_operation.data_reader import DataReader
from experiments.model_loader import ModelLoader, TINYLLAMA_MODEL, MISTRAL7B_MODEL, FALCON40B_MODEL
from experiments.wrappers.exp_data_processing import url_exp_construct_data
from wrapper.url_detection_wrapper import URLDetectionWrapper
import time

INFERENCE_PROMPT = "I will provide some text. Please add a warning before the text without modifying the original " \
                   "content. Do not write any code, but directly return the modified text. Check if the text contains " \
                   "URLs; if it does, list all URLs included in the text. Then, validate whether these URLs are safe, " \
                   "i.e., ensure they are not phishing URLs and are reachable. If any URL is not safe, add a warning " \
                   "before the text and list the unsafe URL. The text is provided below: "


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
    pipe = ModelLoader.load_model(llm_name)
    dataset = url_exp_construct_data()
    new_texts = []
    start_time = time.time()
    for text in dataset:
        new_texts.append(pipe(INFERENCE_PROMPT + text))
    end_time = time.time()
    print(f"Time taken to execute the code: {end_time - start_time} seconds")
    write_to_file(f"{llm_name}_result-{end_time - start_time}.txt", new_texts)
    return new_texts


if __name__ == '__main__':
    # wrapper_test()
    llm_test(TINYLLAMA_MODEL)
