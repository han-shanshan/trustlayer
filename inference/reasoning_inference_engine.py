from datasets import Dataset, load_metric
from transformers import AutoModelForCausalLM
from inference.classification_inference_engine import InferenceEngine
import torch

"""
Reference code: 
https://huggingface.co/docs/transformers/main/en/peft
https://github.com/huggingface/peft/discussions/661
"""


accuracy_metric = load_metric("accuracy", trust_remote_code=True)
precision_metric = load_metric("precision", trust_remote_code=True)
recall_metric = load_metric("recall", trust_remote_code=True)
f1_metric = load_metric("f1", trust_remote_code=True)
# roc_auc_metric = evaluate.load("roc_auc", trust_remote_code=True)


def compute_metrics(labels, predictions, metrics_average="macro"):
    # print(f"labels = {labels}")
    # print(f"predictions = {predictions}")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=metrics_average)
    # roc_auc = roc_auc_metric.compute(references=labels, prediction_scores=probabilities)

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        # "roc_auc": roc_auc["roc_auc"]
    }


class ReasoningInferenceEngine(InferenceEngine):
    def __init__(self, task_name, base_model, model=None, adapter_path=None, inference_config=None,
                 problem_type="single_label_classification"):
        super().__init__(task_name=task_name, base_model=base_model, model=model,
                         adapter_path=adapter_path, inference_config=inference_config, problem_type=problem_type)

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

    @staticmethod
    def get_inference_config(inference_config):
        return None

    def get_hallu_reasoning_prompt_for_fox_instruct(self, input_infor):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. ",
            },
            {
                "role": "user",
                "content": f"According to Question/Dialogue and Knowledge, is there any hallucination in the LLM "
                           f"Response? {input_infor}",
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # chat_template=template,
            # add_generation_prompt=False for training;
            # add_generation_prompt=True for generation/inference
        )
        return prompt

    """
    https://huggingface.co/docs/peft/en/quicktour
    """

    def inference(self, text, text_pair=None, max_new_tokens=100):
        # print(f"device = {next(self.model.parameters()).device}")
        encoded_input = self.tokenizer(text, return_tensors="pt",
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
        if plaintext_result.startswith("No"):
            plaintext_result = "No. "

        # inputs = tokenizer(text, truncation=True, padding=True, max_length=8192,
        #                    return_tensors='pt', return_token_type_ids=False).to(model.device)
        # output = model.generate(**inputs, max_new_tokens=16, output_logits=True,
        #                         return_dict_in_generate=True, output_scores=True)

        # input_ids = self.tokenizer(text, truncation=True, padding=True, max_length=8192,
        #                            return_tensors='pt', return_token_type_ids=False).input_ids
        # outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, output_logits=True,
        #                               return_dict_in_generate=True, output_scores=True)
        # plaintext_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"plaintext result = {plaintext_result}")
        #
        # logits = outputs.logits
        #
        # next_word_logits = logits[0]
        # probs = torch.softmax(next_word_logits, dim=-1)
        # # print("----------")
        # prob_yes = 0
        # prob_no = 0
        # top_k_num = 10
        # while True:
        #     top_probs, top_indices = torch.topk(probs, top_k_num)
        #     prob_list = top_probs[0].tolist()
        #     top_indice = top_indices[0].tolist()
        #     print(f"top_indices = {top_indices}")
        #
        #     for idx in range(len(top_indice)):
        #         next_word = tokenizer.decode([top_indice[idx]])
        #         if str(next_word).lower() == "yes":  # the result might be "Yes", yes", "No", and "no"
        #             prob_yes += prob_list[idx]
        #         if str(next_word).lower() == "no":
        #             prob_no += prob_list[idx]
        #     if prob_yes > 0 or prob_no > 0:
        #         results.append(prob_yes / (prob_yes + prob_no))
        #         break
        #     else:
        #         top_k_num += 10
        #         print("continue...")
        #     if top_k_num >= 50:
        #         results.append(0)
        #         break
        # print(f"first token: {tokenizer.decode([top_indice[0]])}")

        return plaintext_result

    def evaluate(self, plaintext_dataset):
        # Dataset({
        #     features: ['is_hallucination', 'text'],
        #     num_rows: 1000
        # })
        print(f"plaintext dataset = {plaintext_dataset}")

        def tokenize_function(examples):
            inputs = self.tokenizer(examples["text"], truncation=True, padding=True, max_length=8192 + 1,
                                    return_tensors='pt', return_token_type_ids=False)
            return inputs

        # dataset.cleanup_cache_files()
        encoded_dataset = plaintext_dataset.map(tokenize_function, batched=True,
                                                remove_columns=plaintext_dataset.column_names)
        # encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
        print(f"encoded_dataset = {encoded_dataset}")
        encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
        print(f"encoded dataset = {encoded_dataset}")
        # labels = [1 if item == 'Yes' else 0 for item in plaintext_dataset['is_hallucination']]
        # print(f"len labels = {len(labels)}")
        labels = []



        inference_results = []
        with torch.no_grad():
            self.model.eval()
            counter = 0
            for j in range(len(plaintext_dataset)):
                text = plaintext_dataset[j]['text']
                inference_result = self.inference(text=text)

                if not (inference_result.startswith("Yes") or inference_result.startswith("yes") or inference_result.startswith("No") or inference_result.startswith("no")):
                    print(f"inference result = {inference_result}")
                else: 
                    if inference_result.startswith("Yes") or inference_result.startswith("yes"):
                        inference_results.append(1)
                    else: # inference_result.startswith("No") or inference_result.startswith("no"):
                        inference_results.append(0)
                    if plaintext_dataset[j]['is_hallucination'] == "Yes":
                        labels.append(1)
                    else:
                        labels.append(0)
                    
                counter += 1
                if counter % 5 == 0:
                    print(f"{counter} inferences are done. ")
        # print(f"real results = {plaintext_dataset['is_hallucination']}")
        # print(f"inference results = {inference_results}")

        print(compute_metrics(labels=labels, predictions=inference_results))



