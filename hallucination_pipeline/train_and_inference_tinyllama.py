from hallucination_pipeline.hallucination_training_engine import HallucinationTrainingEngine
from multitask_lora.constants import MODEL_NAME_TINYLAMMA, CUSTOMIZED_HALLUCINATION_TASK_NAME
import os
from multitask_lora.inference_engine import InferenceEngine

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

MODEL_NAME = MODEL_NAME_TINYLAMMA
lora_storage_path = MODEL_NAME.split("/")[1]
OUTPUT_DIR = lora_storage_path + "-" + CUSTOMIZED_HALLUCINATION_TASK_NAME

"""
training data: https://huggingface.co/datasets/deepset/prompt-injections
"""

if __name__ == '__main__':
    trainer = HallucinationTrainingEngine(base_model_name=MODEL_NAME)
    trainer.train()
    text = "i'm happy hahaha"

    config = {
        "base_model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "customized_hallucination": {
            "0": 'To fabricate something other than knowledge.',
            "1": 'Recommendations',
            "2": 'AI compiles content responses and can query the warranty time through SN',
            "3": 'Guide the customer to the website to find the return information',
            "4": "Be biased, following the customer's word promises 2 years warranty",
            "5": 'The warranty period was calculated incorrectly, saying that since the customer purchased in November 2021, the warranty period was 12 months',
            "6": 'Compilation of RMA content',
            "7": 'Promise that the customer can also fill in the invoice without invoice. Make up the contents to say that there is a problem with the quantity of the product',
            "8": 'Physical damage is a very common problem on our side, but now there are many cases of non-physical damage that ignore other excuses, such as asking about photos or videos',
            "9": 'Physical damage'
        }
    }

    inference_engine = InferenceEngine(default_task=CUSTOMIZED_HALLUCINATION_TASK_NAME, config=config, problem_type="single_label_classification")
    print(inference_engine.inference([text,text]))
