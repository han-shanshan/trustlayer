from data_operation.data_loader import DataLoader
from data_operation.reasoning_data_preparer import ReasoningDataPreparer
from inference.reasoning_inference_engine import ReasoningInferenceEngine
from utils.constants import FOX_INSTRUCT, HALLUCINATION_REASONING_TASK
from inference.inference_engine import InferenceEngine
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

TASK_NAME = HALLUCINATION_REASONING_TASK
MODEL_NAME = FOX_INSTRUCT

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

# construct_input_output_pairs_for_hallu_detection(self, question, knowledge, llm_answer, log_type,
#                                                          is_hallucination):
# get_input_information(knowledge, llm_answer, log_type, question)

if __name__ == '__main__':
    input_information = ReasoningDataPreparer().get_input_information(
        question="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i "
                 "like him too did you know he also was in Zodiac a crime fiction film.",
        knowledge="Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction "
                  "Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
        log_type="dialogue",
        llm_answer="I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well."
        # llm_answer="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks."
        # output=""  #Yes  # "No, the answer can be deduced from the context. "
    )
    # text = DataLoader().get_llama_prompt_for_hallucination_reasoning_task(input_output_pair["input"],
    #                                                                       input_output_pair["output"])
    print(input_information)
    inference_engine = ReasoningInferenceEngine(task_name=TASK_NAME, base_model=FOX_INSTRUCT,
                                                # adapter_path=FOX_INSTRUCT
                                                adapter_path="./fox_adapters/Fox-Instruct-hallucination_reasoning-2-qa_and_dialogue"
    )

    prompt = inference_engine.get_hallu_reasoning_prompt_for_fox_instruct(input_information)
    print("=============")
    print(f"prompt = {prompt}")
    print(inference_engine.inference(prompt))
