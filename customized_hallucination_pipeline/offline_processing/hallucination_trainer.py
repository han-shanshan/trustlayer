from customized_hallucination_pipeline.offline_processing.hallucination_training_data_processor import \
    HallucinationTrainingDataProcessor
from customized_hallucination_pipeline.test_use_case import HALLUCINATION_INFERENCE_CONFIG
from training.constants import CUSTOMIZED_HALLUCINATION_TASK_NAME, MODEL_NAME_TINYLAMMA, MODEL_NAME_BERT_BASE, \
    FOX_BASE_GPU
from training.trust_inference_engine import TrustInferenceEngine
from training.training_engine import TrainingEngine
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from peft import get_peft_model
from training.config_manager import ConfigManager


class HallucinationTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name):
        super().__init__(base_model_name, task_name=CUSTOMIZED_HALLUCINATION_TASK_NAME)
        self.model_name = self.base_model_name
        if self.base_model_name in [MODEL_NAME_TINYLAMMA, MODEL_NAME_BERT_BASE]:
            self.model_name = self.base_model_name.split("/")[1]
        else:
            self.model_name = "Fox"

    def set_task_type(self, task_name):
        self.task_name = CUSTOMIZED_HALLUCINATION_TASK_NAME

    def set_label_metrics(self):
        self.label_metrics = self.compute_metrics_for_single_label_tasks

    def get_pretrained_model(self, label_dicts, id2label, label2id):
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name,
                                                                              num_labels=len(label_dicts),
                                                                              id2label=id2label,
                                                                              label2id=label2id,
                                                                              load_in_8bit=False
                                                                              )
        return pretrained_model

    def train(self):
        data_processor = HallucinationTrainingDataProcessor()
        dataset, id2labels, label2ids, label_names = data_processor.get_dataset_info(
            file_path="../data/hallucination_cases.xlsx")
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
        output_dir = self.model_name + "-" + self.task_name

        bert_peft_trainer = Trainer(
            model=model,
            args=config_manager.get_training_config(output_dir=output_dir, batch_size=8),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=self.label_metrics,
        )
        bert_peft_trainer.train()
        model.save_pretrained(output_dir + "-final")


MODEL_NAME = MODEL_NAME_TINYLAMMA

if __name__ == '__main__':
    trainer = HallucinationTrainingEngine(base_model_name=FOX_BASE_GPU)
    trainer.train()
    text = "i'm happy hahaha"

    inference_engine = TrustInferenceEngine(default_task=CUSTOMIZED_HALLUCINATION_TASK_NAME,
                                            config=HALLUCINATION_INFERENCE_CONFIG,
                                            problem_type="single_label_classification")
    print(inference_engine.inference([text, text]))
