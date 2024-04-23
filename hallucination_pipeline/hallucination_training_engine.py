from hallucination_pipeline.hallucination_training_data_processor import HallucinationTrainingDataProcessor
from multitask_lora.constants import CUSTOMIZED_HALLUCINATION_TASK_NAME
from multitask_lora.training_engine import TrainingEngine
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from peft import get_peft_model
from multitask_lora.config_manager import ConfigManager


class HallucinationTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name):
        super().__init__(base_model_name, task_name=CUSTOMIZED_HALLUCINATION_TASK_NAME)

    def set_task_type(self, task_name):
        self.task_name = CUSTOMIZED_HALLUCINATION_TASK_NAME

    def set_label_metrics(self):
        self.label_metrics = self.compute_metrics_for_single_label_tasks

    def get_pretrained_model(self, label_dicts, id2label, label2id):
        return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                  num_labels=len(label_dicts),
                                                                  id2label=id2label,
                                                                  label2id=label2id,
                                                                  load_in_8bit=False
                                                                  )
        # return AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
        #                                                           problem_type="multi_label_classification",
        #                                                           num_labels=len(label_dicts),
        #                                                           id2label=id2label,
        #                                                           label2id=label2id,
        #                                                           load_in_8bit=False
        #                                                           )

    # def get_tokenizer(self, model):
    #     tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
    #     if self.base_model_name in [MODEL_NAME_TINYLAMMA]:
    #         tokenizer.pad_token = tokenizer.eos_token
    #         model.config.pad_token_id = model.config.eos_token_id
    #     return tokenizer

    def train(self):
        data_processor = HallucinationTrainingDataProcessor()
        dataset, id2labels, label2ids, label_names = data_processor.get_dataset_info(file_path="data/hallucination_cases.xlsx")
        print(f"id2labels={id2labels}")
        model = self.get_pretrained_model(label_names, id2labels, label2ids)
        tokenizer = self.get_tokenizer(model)
        encoded_dataset = data_processor.process_encoded_datasets(dataset=dataset, tokenizer=tokenizer)

        config_manager = ConfigManager(model=self.base_model_name)
        print("=======start loading metric=========")
        # metric = evaluate.load("accuracy")
        # Define LoRA Config
        model = get_peft_model(model, config_manager.get_lora_config())
        print("=======print_trainable_parameters============")
        model.print_trainable_parameters()  # see % trainable parameters
        # training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=500)
        output_dir = self.base_model_name.split("/")[1] + "-" + self.task_name

        bert_peft_trainer = Trainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=8),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.label_metrics,
        )
        bert_peft_trainer.train()
        model.save_pretrained(output_dir + "-final")
