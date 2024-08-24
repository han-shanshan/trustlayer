from data_operation.reasoning_data_preparer import get_input_information
from inference.reasoning_inference_engine import ReasoningInferenceEngine
from utils.constants import FOX_INSTRUCT, HALLUCINATION_REASONING_TASK
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

TASK_NAME = HALLUCINATION_REASONING_TASK
MODEL_NAME = FOX_INSTRUCT


def a_single_inference():
    input_information = get_input_information(
        question="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i "
                 "like him too did you know he also was in Zodiac a crime fiction film.",
        knowledge="Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction "
                  "Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
        log_type="dialogue",
        # llm_answer="I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well."
        llm_answer="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks."
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


"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""
if __name__ == '__main__':
    from data_operation.reasoning_data_loader import ReasoningDataLoader

    dataset_types = [
        "HaluEval-qa",
        "HaluEval-dialogue",
        "HaluEval-summarization"
    ]
    data_num_dict = {
        "HaluEval-qa": {"train": 8000, "validation": 1000, "test": 1000},
        "HaluEval-dialogue": {"train": 8000, "validation": 1000, "test": 1000},
        "HaluEval-summarization": {"train": 8000, "validation": 1000, "test": 1000},
        # "rag-hallucination1000": {"train": 500, "validation": 20, "test": 0},
    }

    inference_engine = ReasoningInferenceEngine(task_name=TASK_NAME, base_model=FOX_INSTRUCT,
                                                adapter_path="./fox_adapters/Fox-Instruct-hallucination_reasoning-2-qa_and_dialogue"
                                                )
    data_loader = ReasoningDataLoader(tokenizer=inference_engine.tokenizer)

    training_dataset, validation_dataset, test_dataset = data_loader.load_hallucination_data_for_reasoning(
        data_num_dict=data_num_dict, dataset_types=dataset_types)
    print(f"test data = {test_dataset}")
    print(f"============{test_dataset[10]}")

    dataset = data_loader.get_hallu_reasoning_data_for_fox_instruct(training_dataset, validation_dataset, test_dataset,
                                                                    is_inference=True)
    print(f"dataset = {dataset}")
    print(f"dataset[test] = {dataset['test'][0]}")
    # evaluation_dataset = Dataset.from_dict(dataset['test'][:5])
    inference_engine.evaluate(plaintext_dataset=dataset['test'])
