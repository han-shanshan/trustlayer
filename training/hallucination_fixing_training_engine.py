from datasets import load_metric
from training.hallucination_reasoning_training_engine import DataCollatorForCompletionOnlyLM
from transformers import EarlyStoppingCallback, AutoModelForCausalLM
from transformers import Trainer
from training.classification_training_engine import CustomCallback
from training.training_config_manager import TrainingConfigManager
from training.training_engine import TrainingEngine
from utils.constants import FOX_INSTRUCT, FOX_INSTRUCT_REASONING_RESPONSE_TEMPLATE
import evaluate
from datetime import datetime
from data_operation.reasoning_data_loader import ReasoningDataLoader
from utils.util import get_tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

# accuracy = evaluate.load("accuracy")
accuracy_metric = load_metric("accuracy", trust_remote_code=True)
precision_metric = load_metric("precision", trust_remote_code=True)
recall_metric = load_metric("recall", trust_remote_code=True)
f1_metric = load_metric("f1", trust_remote_code=True)
roc_auc_metric = evaluate.load("roc_auc", trust_remote_code=True)


class HallucinationFixingTrainingEngine(TrainingEngine):
    def __init__(self, base_model_name, task_name, config=None):
        super().__init__(base_model_name, task_name, config)
        # self.data_processor = DataProcessor(task_name=self.task_name)
        self.task = task_name
        if self.config is not None:
            if "dataset_types" in self.config:
                self.dataset_types = self.config["dataset_types"]
            else:
                self.dataset_types = None
            if "data_num_dict" in self.config:
                self.data_num_dict = self.config["data_num_dict"]
            else:
                self.data_num_dict = None

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

    def get_training_data(self, idx=None, tokenizer=None):
        data_loader = ReasoningDataLoader(tokenizer=tokenizer)
        training_dataset, validation_dataset, test_dataset = data_loader.load_hallucination_data_for_fixing(
            self.data_num_dict, self.dataset_types)
        if self.base_model_name == FOX_INSTRUCT:
            task_data = data_loader.get_hallu_fixing_data_for_fox_instruct(training_dataset,
                                                                           validation_dataset,
                                                                           test_dataset)
        else:
            raise ValueError(f"Unsupported model name: {self.base_model_name}")
        print(f"task data = {task_data}")
        print(f"sample data = {task_data['train'][0]}")
        return task_data

    def train(self, model, encoded_dataset, batch_size=32, idx=None, tokenizer=None):
        output_dir = self.base_model_name.split("/")[-1] + "-" + self.task_name + "-" + idx
        print(f"encoded_dataset[train] = {encoded_dataset['train']}")
        peft_trainer = Trainer(
            model=model,
            args=TrainingConfigManager.get_training_config(output_dir=output_dir,
                                                           task_name=self.task_name, batch_size=batch_size),
            train_dataset=encoded_dataset["train"],  # training dataset requires column input_ids
            eval_dataset=encoded_dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()],
            data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False,
                                                          response_template=FOX_INSTRUCT_REASONING_RESPONSE_TEMPLATE)
        )
        peft_trainer.train()
        model.save_pretrained(output_dir + "-final")
        return peft_trainer

    def process(self, batch_size=16, rank=0, world_size=1):
        t = str(datetime.now())
        model = self.get_pretrained_model()
        tokenizer = get_tokenizer(self.base_model_name)
        model.resize_token_embeddings(len(tokenizer))
        # model.to(rank)  # Move model to GPU
        # model = DDP(model, device_ids=[rank])  # Wrap model in DDP
        dataset = self.get_training_data(idx=t, tokenizer=tokenizer)


        #
        # train_sampler = DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank)
        # # Create DataLoader
        # train_loader = DataLoader(dataset['train'], sampler=train_sampler, batch_size=8)

        print(f"sample data in training: {dataset['train'][0]}")
        print(f"sample data in validation: {dataset['validation'][0]}")
        print(f"sample data in test: {dataset['test'][0]}")
        encoded_dataset = self.get_encoded_dataset(dataset=dataset, tokenizer=tokenizer)
        self.train(model=model, encoded_dataset=encoded_dataset,
                   batch_size=batch_size, tokenizer=tokenizer, idx=t)
        # self.evaluate(model=trainer.model, dataset=dataset, tokenizer=tokenizer)
