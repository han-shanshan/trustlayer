from data_operation.reasoning_data_preparer import get_input_information
from inference.generation_inference_engine import GenerationInferenceEngine
from inference.reasoning_inference_engine import ReasoningInferenceEngine
from utils.constants import FOX_INSTRUCT, HALLUCINATION_REASONING_TASK, GENERATION_TASK
import os
from datasets import Dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

TASK_NAME = GENERATION_TASK
MODEL_NAME = FOX_INSTRUCT

# def a_single_inference():
#     input_information = get_input_information(
#         question="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i "
#                  "like him too did you know he also was in Zodiac a crime fiction film.",
#         knowledge="Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction "
#                   "Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
#         log_type="dialogue",
#         # llm_answer="I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well."
#         llm_answer="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks."
#         # output=""  #Yes  # "No, the answer can be deduced from the context. "
#     )
#     # text = DataLoader().get_llama_prompt_for_hallucination_reasoning_task(input_output_pair["input"],
#     #                                                                       input_output_pair["output"])
#     print(input_information)
#     inference_engine = GenerationInferenceEngine(task_name=TASK_NAME, base_model=FOX_INSTRUCT,
#                                                 # adapter_path=FOX_INSTRUCT
#                                                 adapter_path="/Fox-1-1.6B-Instruct-v0.1-hallucination_fixing"
#                                                 )
#
#     prompt = inference_engine.get_hallu_reasoning_prompt_for_fox_instruct(input_information)
#     print("=============")
#     print(f"prompt = {prompt}")
#     print(inference_engine.inference(prompt))


def get_consistent_score(data):
    for item in data:
        if item['label'] == 'consistent':
            return item['score']
    return 0



"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""
if __name__ == '__main__':
    from data_operation.reasoning_data_loader import ReasoningDataLoader

    dataset_types = [
        "HaluEval-qa",
        # "HaluEval-dialogue",
        # "HaluEval-summarization"
    ]
    data_num_dict = {
        "HaluEval-qa": {"train": 8000, "validation": 1000, "test": 1000},
        # "HaluEval-dialogue": {"train": 8000, "validation": 1000, "test": 1000},
        # "HaluEval-summarization": {"train": 8000, "validation": 1000, "test": 1000},
        # "rag-hallucination1000": {"train": 500, "validation": 20, "test": 0},
    }

    inference_engine = GenerationInferenceEngine(task_name=TASK_NAME, base_model=FOX_INSTRUCT,
                                                 adapter_path="./Fox-1-1.6B-Instruct-v0.1-hallucination_fixing-2024-10-01 23:12:35.467978-final"
                                                 )
    training_dataset, validation_dataset, test_dataset = inference_engine.data_loader.load_hallucination_data_for_fixing(
        data_num_dict, dataset_types)

    print(f"test data = {test_dataset}")
    # print(f"============{test_dataset[10]}")

    from transformers import pipeline, AutoTokenizer

    halu_detection_pipe = pipeline("text-classification", trust_remote_code=True,
                                   model="vectara/hallucination_evaluation_model",
                                   tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'))
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    scores = []

    # test_dataset = Dataset.from_dict(test_dataset[:2])

    for data_record in test_dataset:
        res = inference_engine.inference(record=data_record)
        # print(f"data_record['right_answer'] = {data_record['right_answer']}")
        # print(f"res = {res}")
        input_pairs = prompt.format(text1=data_record['right_answer'], text2=res)
        full_scores = halu_detection_pipe(input_pairs, top_k=None)
        score = get_consistent_score(full_scores)  # consistent score

        print(f"full score = {full_scores}, score = {score}")
        scores.append(score)

    print(scores)
    count = sum(1 for num in scores if num > 0.5)
    print(count)
