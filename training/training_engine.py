class TrainingEngine:
    def __init__(self, base_model_name, task_name, config=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.label_metrics = None
        self.config = config

    def get_training_data(self):
        pass

    def get_pretrained_model(self):
        pass

    @staticmethod
    def get_tokenizer(model, base_model_name):
        pass

    def get_encoded_dataset(self, dataset, tokenizer):
        pass

    def evaluate(self):
        pass

    def train(self, model, encoded_dataset):
        pass

    def process(self):
        pass
