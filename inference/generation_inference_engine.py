from transformers import AutoModelForCausalLM
from data_operation.reasoning_data_loader import ReasoningDataLoader
from inference.classification_inference_engine import InferenceEngine
import torch


class GenerationInferenceEngine(InferenceEngine):
    def __init__(self, task_name, base_model, model=None, adapter_path=None, inference_config=None,
                 problem_type="single_label_classification"):
        super().__init__(task_name=task_name, base_model=base_model, model=model,
                         adapter_path=adapter_path, inference_config=inference_config, problem_type=problem_type)
        self.data_loader = ReasoningDataLoader(tokenizer=self.tokenizer)

    def get_model(self, base_model=None, adapter_path=None, model=None):
        if model is not None:
            return model
        if adapter_path is not None:
            model_path = adapter_path
        elif base_model is not None:
            model_path = base_model
        else:
            raise ValueError("Base_model is None and adapter_path is None")
        return AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=False,
                                                    torch_dtype=torch.float32, trust_remote_code=True)

    def get_generation_prompt_for_fixing(self, example):
        prompt = self.data_loader.apply_hallu_fixing_template(question=example['question'],
                                                              knowledge=example['knowledge'],
                                                              hallucinated_answer=example['hallucinated_answer'],
                                                              hallucination_reason=example['hallucination_reason'],
                                                              correct_answer=example['right_answer'],
                                                              log_type=example["task_type"],
                                                              is_inference=True)
        return prompt

    def inference(self, record, text_pair=None, max_new_tokens=100):
        prompt = self.get_generation_prompt_for_fixing(record)
        # print(f"prompt = {prompt}")
        encoded_input = self.tokenizer(prompt, return_tensors="pt",
                                       truncation=True, padding=True, max_length=8192, return_token_type_ids=False)
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}

        outputs = self.model.generate(input_ids=encoded_input['input_ids'], max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      # temperature=0,  # Adjust temperature for more controlled randomness
                                      # top_p=0.9, early_stopping=True
                                      attention_mask=encoded_input['attention_mask'])

        # print(f"outputs[0] = {outputs[0]}")
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"result = {result}")
        plaintext_result = result.split("<|assistant|>")[-1]  # .split("\n")[0]
        plaintext_result = plaintext_result.strip().split("\n")[0]
        # if plaintext_result.startswith("No"):
        #     plaintext_result = "No. "

        return plaintext_result

    # def evaluate(self, plaintext_dataset):
    #     # Dataset({
    #     #     features: ['is_hallucination', 'text'],
    #     #     num_rows: 1000
    #     # })
    #     print(f"plaintext dataset = {plaintext_dataset}")
    #
    #     def tokenize_function(examples):
    #         inputs = self.tokenizer(examples["text"], truncation=True, padding=True, max_length=8192 + 1,
    #                                 return_tensors='pt', return_token_type_ids=False)
    #         return inputs
    #
    #     # dataset.cleanup_cache_files()
    #     encoded_dataset = plaintext_dataset.map(tokenize_function, batched=True,
    #                                             remove_columns=plaintext_dataset.column_names)
    #     # encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
    #     print(f"encoded_dataset = {encoded_dataset}")
    #     encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
    #     print(f"encoded dataset = {encoded_dataset}")
    #     # labels = [1 if item == 'Yes' else 0 for item in plaintext_dataset['is_hallucination']]
    # print(f"len labels = {len(labels)}")
    # labels = []
    #
    # inference_results = []
    # with torch.no_grad():
    #     self.model.eval()
    #     counter = 0
    #     for j in range(len(plaintext_dataset)):
    #         text = plaintext_dataset[j]['text']
    #         inference_result = self.inference(text=text)
    #
    #         if not (inference_result.startswith("Yes") or inference_result.startswith(
    #                 "yes") or inference_result.startswith("No") or inference_result.startswith("no")):
    #             print(f"inference result = {inference_result}")
    #         else:
    #             if inference_result.startswith("Yes") or inference_result.startswith("yes"):
    #                 inference_results.append(1)
    #             else:  # inference_result.startswith("No") or inference_result.startswith("no"):
    #                 inference_results.append(0)
    #             if plaintext_dataset[j]['is_hallucination'] == "Yes":
    #                 labels.append(1)
    #             else:
    #                 labels.append(0)
    #
    #         counter += 1
    #         if counter % 5 == 0:
    #             print(f"{counter} inferences are done. ")
    # # print(f"real results = {plaintext_dataset['is_hallucination']}")
    # # print(f"inference results = {inference_results}")
    #
    # print(compute_metrics(labels=labels, predictions=inference_results))
