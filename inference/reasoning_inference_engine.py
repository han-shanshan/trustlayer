from transformers import AutoModelForCausalLM
from inference.inference_engine import InferenceEngine
import torch

"""
Reference code: 
https://huggingface.co/docs/transformers/main/en/peft
https://github.com/huggingface/peft/discussions/661
"""


class ReasoningInferenceEngine(InferenceEngine):
    def __init__(self, task_name, base_model, adapter_path=None, inference_config=None,
                 problem_type="single_label_classification"):
        super().__init__(task_name, base_model, adapter_path, inference_config, problem_type)

    def get_model(self, model_path):
        return AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=False,
                                                    torch_dtype=torch.float32, trust_remote_code=True)

    @staticmethod
    def get_inference_config(inference_config):
        return None

    """
    https://huggingface.co/docs/peft/en/quicktour
    """
    def inference(self, text, text_pair=None, max_new_tokens=100):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        plaintext_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return plaintext_result

    def evaluate(self, dataset):
        pass
