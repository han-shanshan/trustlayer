import warnings
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, EarlyStoppingCallback, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer
from peft import get_peft_model
import torch
from data_operation.reasoning_data_loader import ReasoningDataLoader
from training.classification_training_engine import CustomCallback
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine
from utils.constants import FOX_INSTRUCT, FOX_INSTRUCT_REASONING_RESPONSE_TEMPLATE
from data_operation.data_processor import DataProcessor
import evaluate
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# accuracy = evaluate.load("accuracy")
accuracy_metric = load_metric("accuracy", trust_remote_code=True)
precision_metric = load_metric("precision", trust_remote_code=True)
recall_metric = load_metric("recall", trust_remote_code=True)
f1_metric = load_metric("f1", trust_remote_code=True)
roc_auc_metric = evaluate.load("roc_auc", trust_remote_code=True)


# tokenizer = AutoTokenizer.from_pretrained(FOX_BASE_GPU)
# tokenizer.pad_token = tokenizer.eos_token
#
# id1 = tokenizer.encode('Yes', add_special_tokens=False)[0]
# id2 = tokenizer.encode('yes', add_special_tokens=False)[0]
# id3 = tokenizer.encode('No', add_special_tokens=False)[0]
# id4 = tokenizer.encode('no', add_special_tokens=False)[0]
#
# valid_token_ids = [id1, id2, id3, id4]

# Adapted from https://github.com/huggingface/trl/blob/01c4a35928f41ba25b1d0032a085519b8065c843/trl/trainer/utils.py#L56
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
            self,
            tokenizer,
            response_template: str,
            ignore_index: int = -100,
            mlm: bool = True,
            mlm_probability: float = 0.15,
            pad_to_multiple_of: Optional[int] = None,
            tf_experimental_compile: bool = False,
            return_tensors: str = "pt"
    ):
        super().__init__(
            tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            tf_experimental_compile=tf_experimental_compile,
            return_tensors=return_tensors
        )

        self.ignore_index = ignore_index

        if len(response_template) == 0:
            raise ValueError(f"{type(self).__name__} requires a non-empty `response_template`.")

        # The prompt ends with the response template. We encode this and then try to find it in the
        # sequence of tokens.
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

        # See https://github.com/huggingface/trl/pull/622
        # See https://github.com/huggingface/trl/issues/598
        # See https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate
        # Some tokenizers such as "GPT2Tokenizer" and "Llama2Tokenizer" tokenize input string differently
        # depending on the context. Below are fallback solutions
        self.response_template_ctx = f"\n{response_template}"
        self.response_ctx_token_ids = self.tokenizer.encode(self.response_template_ctx, add_special_tokens=False)[2:]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                        self.response_token_ids
                        == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx
                    break
            else:
                # Fallback to `response_ctx_token_ids` for tokenizers that requires the input in
                # context (e.g. "GPT2Tokenizer" and "Llama2Tokenizer")
                for idx in np.where(batch["labels"][i] == self.response_ctx_token_ids[0])[0]:
                    if (
                            self.response_ctx_token_ids
                            == batch["labels"][i][idx: idx + len(self.response_ctx_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx
                        break

                else:
                    input_ids = batch['input_ids'][i][batch['attention_mask'][i] > 0].tolist()
                    # print(f"input_ids type = {type(input_ids)}, len = {len(input_ids)}, first element = {input_ids[0]}, type of first = {type(input_ids[0])}")
                    # print(f"input ids = {input_ids}")
                    warnings.warn(
                        f"{type(self).__name__} Could not find response key `{self.response_template}` in the"
                        f" following instance: ```{self.tokenizer.decode(input_ids)}```"
                        f" This instance will be ignored in loss calculation."
                        f" Note, if this happens often, consider increasing the `max_seq_length`."
                    )

                    # set to the max length of the current sample
                    response_token_ids_start_idx = len(batch["labels"][i])

            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response template
            batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch


class HallucinationReasoningTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name, task_name, config=None):
        super().__init__(base_model_name, task_name, config)
        self.data_processor = DataProcessor(task_name=self.task_name)
        if self.config is not None:
            if "dataset_types" in self.config:
                self.dataset_types = self.config["dataset_types"]
            else:
                self.dataset_types = None
            if "data_num_dict" in self.config:
                self.data_num_dict = self.config["data_num_dict"]
            else:
                self.data_num_dict = None

    def get_training_data(self, idx=None, tokenizer=None):
        data_loader = ReasoningDataLoader(tokenizer=tokenizer)
        training_dataset, validation_dataset, test_dataset = data_loader.load_hallucination_data_for_reasoning(
            self.data_num_dict, self.dataset_types)
        if self.base_model_name == FOX_INSTRUCT:
            task_data = data_loader.get_hallu_reasoning_data_for_fox_instruct(training_dataset,
                                                                              validation_dataset,
                                                                              test_dataset)
        else:
            task_data = data_loader.get_hybrid_hallucination_data_for_fox_base(training_dataset,
                                                                               validation_dataset,
                                                                               test_dataset)
        print(f"task data = {task_data}")
        print(f"sample data = {task_data['train'][0]}")
        return task_data

    def get_pretrained_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, load_in_8bit=False,
                                                     # device_map="auto",
                                                     torch_dtype=torch.float32,
                                                     trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id
        model.enable_input_require_grads()
        # this line solves the bug: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # model.config.use_cache = False

        model = get_peft_model(model, TrainingConfigManager.get_lora_config(model_name=self.base_model_name))
        model.print_trainable_parameters()  # see % trainable parameters

        return model

    def get_encoded_dataset(self, dataset, tokenizer):
        print(f"before enocding = {dataset}")

        def tokenize_function(examples):
            inputs = tokenizer(examples["text"], truncation=True, padding=True, max_length=8192 + 1,
                               return_tensors='pt')
            return inputs

        # dataset.cleanup_cache_files()
        encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
        print(f"encoded_dataset = {encoded_dataset}")
        return encoded_dataset

    def evaluate(self, model=None, dataset=None, tokenizer=None):
        results = []
        real_results = []
        with torch.no_grad():
            model.eval()

            for j in range(5):
                # for j in range(len(dataset["test"])):
                # text = DataLoader.get_llama_prompt_for_hallucination_reasoning_task(dataset["test"][j]['text'], "")
                text = dataset["test"][j]['text']
                print(f"text = {text}")
                self.evaluate_one_input(model, results, text, tokenizer)
                print(f"desired output = {str(dataset['test'][j]['output']).lower()}, prob = {results[j]}")

                if str(dataset["test"][j]['output']).lower() == "yes":
                    real_results.append(1)
                else:
                    real_results.append(0)

    @staticmethod
    def evaluate_one_input(model, results, text, tokenizer):
        inputs = tokenizer(text, truncation=True, padding=True, max_length=8192,
                           return_tensors='pt', return_token_type_ids=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, output_logits=True,
                                return_dict_in_generate=True, output_scores=True)
        logits = output.logits
        next_word_logits = logits[0]
        probs = torch.softmax(next_word_logits, dim=-1)
        prob_yes = 0
        prob_no = 0
        top_k_num = 10
        while True:
            top_probs, top_indices = torch.topk(probs, top_k_num)
            prob_list = top_probs[0].tolist()
            top_indice = top_indices[0].tolist()
            print(f"top_indices = {top_indices}")

            for idx in range(len(top_indice)):
                next_word = tokenizer.decode([top_indice[idx]])
                if str(next_word).lower() == "yes":  # the result might be "Yes", yes", "No", and "no"
                    prob_yes += prob_list[idx]
                if str(next_word).lower() == "no":
                    prob_no += prob_list[idx]
            if prob_yes > 0 or prob_no > 0:
                results.append(prob_yes / (prob_yes + prob_no))
                break
            else:
                top_k_num += 10
                print("continue...")
            if top_k_num >= 50:
                results.append(0)
                break
        print(f"first token: {tokenizer.decode([top_indice[0]])}")

    def train(self, model, encoded_dataset, batch_size=32, idx=None, tokenizer=None):
        output_dir = self.base_model_name.split("/")[-1] + "-" + self.task_name + "-" + idx
        peft_trainer = Trainer(
            model=model,
            args=TrainingConfigManager.get_training_config(output_dir=output_dir,
                                                           task_name=self.task_name, batch_size=batch_size),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()],
            data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False,
                                                          response_template=FOX_INSTRUCT_REASONING_RESPONSE_TEMPLATE)
            # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        peft_trainer.train()
        model.save_pretrained(output_dir + "-final")
        return peft_trainer

    @staticmethod
    def get_tokenizer(model, base_model_name):
        tokenizer = AutoTokenizer.from_pretrained(FOX_INSTRUCT, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def process(self, batch_size=16):
        t = str(datetime.now())
        model = self.get_pretrained_model()
        tokenizer = self.get_tokenizer(model, self.base_model_name)
        model.resize_token_embeddings(len(tokenizer))
        dataset = self.get_training_data(idx=t, tokenizer=tokenizer)
        encoded_dataset = self.get_encoded_dataset(dataset=dataset, tokenizer=tokenizer)
        trainer = self.train(model=model, encoded_dataset=encoded_dataset,
                             batch_size=batch_size, tokenizer=tokenizer, idx=t)
        self.evaluate(model=trainer.model, dataset=dataset, tokenizer=tokenizer)
