from data_operation.data_loader import DataLoader
from inference.reasoning_inference_engine import ReasoningInferenceEngine
from utils.constants import FOX, HALLUCINATION_EXPLANATION_TASK_NAME
from inference.inference_engine import InferenceEngine
import os
from datasets import load_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

TASK_NAME = HALLUCINATION_EXPLANATION_TASK_NAME
MODEL_NAME = FOX  # "google-bert/bert-base-uncased"

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    input_output_pair = DataLoader().construct_input_output_pairs_for_hall_detection(
        question="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i "
                 "like him too did you know he also was in Zodiac a crime fiction film.",
        knowledge="Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction "
                  "Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
        llm_answer="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.",
        output=""  #Yes  # "No, the answer can be deduced from the context. "
    )
    text = DataLoader().get_llama_prompt_for_hallucination_reasoning_task(input_output_pair["input"],
                                                                          input_output_pair["output"])
    print(text)
    inference_engine = ReasoningInferenceEngine(task_name=TASK_NAME, base_model=FOX,
                                                adapter_path="./fox_adapters/Fox-1-1.6B-hallucination_explanation"
                                                             "-2024-06-25 17:25:08.253047-final")
    print(inference_engine.inference(text))
