from nemoguardrails import LLMRails, RailsConfig
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from training.training_engine import compute_metrics


def nemo_input_rail_test(dataset):
    config = RailsConfig.from_path("./nemo_config/")
    rails = LLMRails(config)
    labels = dataset["label"]
    probabilities = []
    predictions = []
    counter = 0
    for text in dataset["text"]:
        response = rails.generate(text)
        # print(f"{text}\n response = {response}\n----------------------------\n")
        if response == "ALLOW":
            predictions.append(0)
        elif response.strip() in ["DENY", "I'm sorry, I can't respond to that."]:
            predictions.append(1)
        probabilities.append(0)
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} done. Current prediction: {predictions[-1]}, text = {text}")

    metrics = compute_metrics(labels, predictions, probabilities)
    metrics.pop('roc_auc')   # probabilities are fake; just to make the parameters suitable for compute_metrics
    print(f"metrics = {metrics}")


if __name__ == '__main__':
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    dataset = prepare_toxic_chat_test_data()
    nemo_input_rail_test(dataset)
    print(f"Nemo Experiments Done.")
