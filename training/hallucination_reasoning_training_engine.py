from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from data_operation.data_loader import DataLoader
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainerCallback, \
    TrainerState, TrainerControl, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer
from peft import get_peft_model
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine, CustomCallback
from utils.constants import GIBBERISH_TASK_NAME, UNSAFE_PROMPT_TASK_NAME, HALLUCINATION_TASK_NAME, \
    TOXICITY_TASK_NAME, MODEL_NAME_TINYLAMMA, FOX_BASE_GPU, SEMANTIC_TASK_NAME, TOPIC_TASK_NAME, \
    CUSTOMIZED_HALLUCINATION_TASK_NAME, ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME, \
    HALLUCINATION_EXPLANATION_TASK_NAME
from data_operation.data_processor import DataProcessor
import evaluate
from utils.file_operations import write_hf_dataset_to_csv
from scipy.special import expit as sigmoid
from datetime import datetime
import nn

# accuracy = evaluate.load("accuracy")
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")
roc_auc_metric = evaluate.load("roc_auc")

tokenizer = AutoTokenizer.from_pretrained(FOX_BASE_GPU)

id1 = tokenizer.encode('Yes', add_special_tokens=False)[0]
id2 = tokenizer.encode('yes', add_special_tokens=False)[0]
id3 = tokenizer.encode('No', add_special_tokens=False)[0]
id4 = tokenizer.encode('no', add_special_tokens=False)[0]

valid_token_ids = [id1, id2, id3, id4]


class HallucinationReasoningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        logits = outputs.logits
        # Extract logits corresponding to "Yes", "yes", "No", and "no" for the first word
        valid_logits = logits[:, 0, valid_token_ids]
        targets = torch.tensor([[0.045, 0.005, 0.9, 0.05]], device=logits.device).repeat(logits.shape[0], 1)
        loss_fct = nn.KLDivLoss(reduction='batchmean')  # Compute multi-class cross-entropy loss
        log_probs = nn.LogSoftmax(dim=-1)(valid_logits)
        loss_for_classification = loss_fct(log_probs, targets)

        loss += loss_for_classification

        return (loss, outputs) if return_outputs else loss


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
                                                    torch_dtype=torch.float32, trust_remote_code=True)
        # encoded_dataset = data_processor.process_encoded_datasets(dataset=dataset, tokenizer=tokenizer)
        # print(f"encoded_dataset in training: {encoded_dataset}")
        config_manager = TrainingConfigManager(self.task_name, self.base_model_name, config=self.config)
        model = get_peft_model(model, config_manager.get_lora_config())
        model.print_trainable_parameters()  # see % trainable parameters

        peft_trainer = HallucinationReasoningTrainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=32),
            train_dataset=dataset["train"],  # training dataset requires column input_ids
            eval_dataset=dataset["validation"],
            compute_metrics=self.label_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        peft_trainer.train()
        # test_results = peft_trainer.evaluate(eval_dataset=encoded_dataset["test"])
        # print("Test Results with hybrid test data:", test_results)
        model.save_pretrained(output_dir + "-final")

        results = []
        real_results = []

        for j in range(len(dataset["test"])):
            text = prompter.generate_prompt(
                "Is there any fraud order in the following orders? Answer with 'yes' or 'no' only. Explain why",
                dataset["test"][j]['input'])
            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)  # .to(device)

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
