from typing import Dict, List, Union, Any, Optional
import warnings
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
# from trl import DataCollatorForCompletionOnlyLM
from data_operation.data_loader import DataLoader
import numpy as np
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, EvalPrediction
from transformers import Trainer
from peft import get_peft_model
import torch
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine, CustomCallback, compute_metrics
from utils.constants import FOX_BASE_GPU, EXPLANATION_RESPONSE_TEMPLATE
from data_operation.data_processor import DataProcessor
import evaluate
from utils.file_operations import write_hf_dataset_to_csv
from datetime import datetime
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# accuracy = evaluate.load("accuracy")
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")
roc_auc_metric = evaluate.load("roc_auc")

# tokenizer = AutoTokenizer.from_pretrained(FOX_BASE_GPU)
# tokenizer.pad_token = tokenizer.eos_token
#
# id1 = tokenizer.encode('Yes', add_special_tokens=False)[0]
# id2 = tokenizer.encode('yes', add_special_tokens=False)[0]
# id3 = tokenizer.encode('No', add_special_tokens=False)[0]
# id4 = tokenizer.encode('no', add_special_tokens=False)[0]
#
# valid_token_ids = [id1, id2, id3, id4]


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
                    input_ids = batch['labels'][i][batch['attention_mask'][i] > 0].tolist()

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

    def set_task_type(self, task_name):
        self.task_name = task_name

    def get_tokenizer(self, base_model_name):
        return

    def train(self, desired_total_data_n=None,batch_size=16):
        t = str(datetime.now())
        data_processor = DataProcessor(task_name=self.task_name)
        dataset, _, _, _ = data_processor.get_dataset(dataset_types=self.dataset_types,
                                                      data_num_dict=self.data_num_dict,
                                                      desired_total_data_n=desired_total_data_n)
        write_hf_dataset_to_csv(dataset['train'], f"{self.task_name}_train_data_{t}.csv")
        write_hf_dataset_to_csv(dataset['validation'], f"{self.task_name}_validation_data_{t}.csv")
        print(f"dataset in training: {dataset}")
        print(f"sample data = {dataset['train'][0]}")
        output_dir = self.base_model_name.split("/")[-1] + "-" + self.task_name + "-" + t
        write_hf_dataset_to_csv(dataset['test'], f"{self.task_name}_test_data_{t}.csv")
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, load_in_8bit=False,
                                                     # device_map="auto",
                                                     torch_dtype=torch.float32,
                                                     trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id
        print(f"dataset = {dataset}")
        print(f"{dataset['train'][0]}")
        print(f"{dataset['train'][1]}")
        print(f"{dataset['train'][2]}")
        exit(0)

        tokenizer = AutoTokenizer.from_pretrained(FOX_BASE_GPU, use_fast = False)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            inputs = tokenizer(examples["text"], truncation=True, padding=True, max_length=8192+1,
                               return_tensors='pt')
            return inputs
        
        # dataset.cleanup_cache_files()
        encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        encoded_dataset = encoded_dataset.filter(lambda x: len(x["input_ids"]) <= 8192)
        print(encoded_dataset)

        config_manager = TrainingConfigManager(self.task_name, self.base_model_name, config=self.config)
        model.enable_input_require_grads()  # this line solves this bug: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # model.config.use_cache = False
        model = get_peft_model(model, config_manager.get_lora_config())
        model.print_trainable_parameters()  # see % trainable parameters
        model.resize_token_embeddings(len(tokenizer))
        print(f"dataset = {dataset}")


        peft_trainer = Trainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=batch_size),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            # tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()],
            data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, response_template=EXPLANATION_RESPONSE_TEMPLATE)
            # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        print(f"EXPLANATION_RESPONSE_TEMPLATE = {EXPLANATION_RESPONSE_TEMPLATE}")

        peft_trainer.train()
        model.save_pretrained(output_dir + "-final")


        results = []
        real_results = []
        probabilities = []
        with torch.no_grad():
            model.eval()

            for j in range(len(dataset["test"])):
                text = dataset["test"][j]['text']
                print(f"text = {text}")
                inputs = tokenizer(text, truncation=True, padding=True, max_length=8192,
                                    return_tensors='pt', return_token_type_ids=False).to(model.device)
                output = model.generate(**inputs, max_new_tokens=16, output_logits=True, return_dict_in_generate=True, output_scores=True)
        
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
                    # print(f"top_indices = {top_indices}")

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

                probabilities.append(results[j])

                print(
                    f"first token: {tokenizer.decode([top_indice[0]])}, desired output = {str(dataset['test'][j]['output']).lower()}, prob = {results[j]}")

                if str(dataset["test"][j]['output'].strip()).lower() == "yes":
                    real_results.append(1)
                else:
                    real_results.append(0)
            print(f"real_results = {real_results}")
            print(f"results = {results}")
            print(f"probabilities = {probabilities}")
            metrics = compute_metrics(labels=real_results, predictions=results, probabilities=probabilities)
            print(f"metrics = {metrics}")

