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
    def inference(self, text, text_pair=None):
        encoding = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**encoding)
        print(outputs)
        return outputs

    def evaluate(self, dataset):
        pass
        # labels = dataset["label"]
        # texts = dataset["text"]
        # # dataset = dataset.remove_columns('label')
        # predictions = []
        # probabilities = []
        # counter = 0
        # for text in texts:
        #     encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=516,
        #                               return_tensors="pt")
        #     encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        #     # print(f"encoding =========================== {encoding}")
        #     outputs = self.model(**encoding)
        #     logits = outputs.logits
        #     predicted_label_idx = torch.argmax(logits, dim=-1).item()
        #     probability = sigmoid(logits[:, 1].cpu().detach()).item()
        #     predictions.append(predicted_label_idx)
        #     # print(f"predicted_label_idx = {predicted_label_idx}")
        #     probabilities.append(probability)
        #
        #     if counter % 100 == 0:
        #         print(f"label = {predicted_label_idx}, real label = {labels[counter]}, text = {text}")
        #     counter += 1
        #
        # metrics = compute_metrics(labels, predictions, probabilities)
        # print(f"metrics = {metrics}")
