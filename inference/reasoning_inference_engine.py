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
        encoded_input = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, padding=True, max_length=8192, return_token_type_ids=False)
        input_ids = encoded_input.input_ids
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      temperature=0,  # Adjust temperature for more controlled randomness
                                      top_p=0.9,
                                      attention_mask=encoded_input.attention_mask,
                                      early_stopping=True)
        # print(f"outputs = {outputs}")

        # print(f"outputs[0] = {outputs[0]}")
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"result = {result}")
        plaintext_result = result.split("<|assistant|>")[-1]# .split("\n")[0]
        plaintext_result = plaintext_result.strip().split("\n")[0]
        print(f"plaintext result 2 = {plaintext_result}")
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

    def evaluate(self, dataset):
        pass
