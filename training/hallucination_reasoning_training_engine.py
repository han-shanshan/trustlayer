from typing import Dict, List, Union, Any, Optional
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

from data_operation.data_loader import DataLoader
import numpy as np
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction
from transformers import Trainer
from peft import get_peft_model
import torch
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine, CustomCallback
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

tokenizer = AutoTokenizer.from_pretrained(FOX_BASE_GPU)
tokenizer.pad_token = tokenizer.eos_token

id1 = tokenizer.encode('Yes', add_special_tokens=False)[0]
id2 = tokenizer.encode('yes', add_special_tokens=False)[0]
id3 = tokenizer.encode('No', add_special_tokens=False)[0]
id4 = tokenizer.encode('no', add_special_tokens=False)[0]

valid_token_ids = [id1, id2, id3, id4]


class HallucinationReasoningTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init,
                         compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        self.Yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.No_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # logits = outputs.logits
        # # Extract logits corresponding to "Yes", "yes", "No", and "no" for the first word
        # valid_logits = logits[:, 0, valid_token_ids]
        # targets = torch.tensor([[0.045, 0.005, 0.9, 0.05]], device=logits.device).repeat(logits.shape[0], 1)
        # loss_fct = nn.KLDivLoss(reduction='batchmean')  # Compute multi-class cross-entropy loss
        # log_probs = nn.LogSoftmax(dim=-1)(valid_logits)
        # loss_for_classification = loss_fct(log_probs, targets)
        #
        # loss += loss_for_classification
        # Custom behavior for preferring "Yes" or "No" after response_template
        if labels is not None:
            response_template_ids = self.data_collator.response_token_ids
            response_token_ids_end_idx = None

            for i in range(len(labels)):
                for idx in np.where(labels[i] == response_template_ids[0])[0]:
                    if response_template_ids == labels[i][idx: idx + len(response_template_ids)].tolist():
                        response_token_ids_end_idx = idx + len(response_template_ids)

                if response_token_ids_end_idx is not None:
                    next_token_id = labels[i][response_token_ids_end_idx].item()
                    if next_token_id != self.yes_token_id and next_token_id != self.Yes_token_id \
                            and next_token_id != self.no_token_id and next_token_id != self.No_token_id:
                        loss += 0.2  # Add a penalty to the loss

        return (loss, outputs) if return_outputs else loss

        # if labels is not None:
        #     response_template_ids = self.data_collator.response_token_ids
        #     yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        #     no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        #
        #     # Cross-entropy loss for the first token after response_template
        #     cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
        #
        #     for i in range(len(labels)):
        #         response_token_ids_end_idx = None
        #         for idx in np.where(labels[i] == response_template_ids[0])[0]:
        #             if response_template_ids == labels[i][idx: idx + len(response_template_ids)].tolist():
        #                 response_token_ids_end_idx = idx + len(response_template_ids)
        #                 break
        #
        #         if response_token_ids_end_idx is not None:
        #             next_token_logits = outputs["logits"][i, response_token_ids_end_idx, :]
        #             target_labels = torch.tensor([yes_token_id, no_token_id]).to(next_token_logits.device)
        #             target_probs = torch.zeros_like(next_token_logits).scatter_(0, target_labels,
        #                                                                         1.0 / len(target_labels))
        #             penalty_loss = cross_entropy_loss_fn(next_token_logits.unsqueeze(0), target_probs.unsqueeze(0))
        #
        #             loss += penalty_loss


class DataCollatorForHallucinationExplanationLM(DataCollatorForLanguageModeling):
    """
    https://github.com/huggingface/trl/blob/84156f179f91f519e48185414391d040112f2d34/trl/trainer/utils.py#L175
    """

    def __init__(
            self,
            response_template: Union[str, List[int]] = None,
            instruction_template: Optional[Union[str, List[int]]] = None,
            *args,
            mlm: bool = False,
            ignore_index: int = -100,
            **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                        self.response_token_ids
                        == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                batch["labels"][i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
        return batch


class HallucinationReasoningTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name, task_name, config=None):
        super().__init__(base_model_name, task_name, config)

    def set_task_type(self, task_name):
        self.task_name = task_name

    def get_tokenizer(self, base_model_name):
        return

    def train(self, desired_total_data_n=None):
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

        def tokenize_function(examples):
            inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512,
                               return_tensors='pt')
            return inputs

        encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        config_manager = TrainingConfigManager(self.task_name, self.base_model_name, config=self.config)
        model.enable_input_require_grads()  # this line solves this bug: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # model.config.use_cache = False
        model = get_peft_model(model, config_manager.get_lora_config())
        model.print_trainable_parameters()  # see % trainable parameters
        print(f"dataset = {dataset}")

        print(f"config = {config_manager.get_training_config(output_dir=output_dir, batch_size=2)}")

        peft_trainer = HallucinationReasoningTrainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=2),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()],
            data_collator=DataCollatorForHallucinationExplanationLM(tokenizer=tokenizer, mlm=False,
                                                                    response_template=EXPLANATION_RESPONSE_TEMPLATE),
        )

        peft_trainer.train()
        # test_results = peft_trainer.evaluate(eval_dataset=encoded_dataset["test"])
        # print("Test Results with hybrid test data:", test_results)
        model.save_pretrained(output_dir + "-final")

        results = []
        real_results = []

        for j in range(len(dataset["test"])):
            text = DataLoader.get_llama_prompt_for_hallucination_reasoning_task(dataset["test"][j]['text'], "")
            inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512,
                               return_tensors='pt', return_token_type_ids=False)  # .to(device)

            # outputs = model.generate(**inputs, top_k=1, max_new_tokens=1, pad_token_id=11)
            next_word_logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(next_word_logits, dim=-1)

            prob_yes = 0
            prob_no = 0
            top_k_num = 10
            while True:
                top_probs, top_indices = torch.topk(probs, top_k_num)
                prob_list = top_probs[0].tolist()
                top_indice = top_indices[0].tolist()

                for idx in range(len(top_indice)):
                    next_word = tokenizer.decode(top_indice[idx])
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

            print(
                f"first token: {tokenizer.decode(top_indice[0])}, desired output = {str(dataset['test'][j]['output']).lower()}, prob = {results[j]}")

            if str(dataset["test"][j]['output']).lower() == "yes":
                real_results.append(1)
            else:
                real_results.append(0)
