from datasets import load_dataset
from nemoguardrails import LLMRails, RailsConfig
from training.training_engine import compute_metrics


def nemo_input_rail_test(dataset):
    config = RailsConfig.from_path("./nemo_config/")
    rails = LLMRails(config)
    labels = dataset["train"]["label"][:5]
    probabilities = []
    predictions = []
    for text in dataset["train"]["text"][:5]:
        response = rails.generate(text)
        print(f"{text}\n response = {response}\n----------------------------\n")
        if response == "ALLOW":
            predictions.append(0)
        elif response.strip() in ["DENY", "I'm sorry, I can't respond to that."]:
            predictions.append(1)
        probabilities.append(0)

    metrics = compute_metrics(labels, predictions, probabilities)
    metrics.pop('roc_auc')   # probabilities are fake; just to make the parameters suitable for compute_metrics
    print(f"metrics = {metrics}")


if __name__ == '__main__':
    test_data = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    print(test_data)
    nemo_input_rail_test(test_data)
