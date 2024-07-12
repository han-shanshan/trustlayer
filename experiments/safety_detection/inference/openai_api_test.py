from data_operation.data_file_operator import FileOperator
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
import requests

from training.classification_training_engine import compute_metrics


def openai_test(dataset):
    url = "https://api.openai.com/v1/moderations"
    key = FileOperator().read_openai_apikey()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    labels = dataset["label"]
    probabilities = []
    predictions = []
    counter = 0
    for text in dataset["text"]:
        response = requests.post(url, json={"input": text}, headers=headers)
        response = response.json()
        score = 0
        if response["results"][0]["flagged"]:
            prediction = 1
        else:
            prediction = 0
        for s in response["results"][0]["category_scores"].values():
            if s > score:
                score = s
        predictions.append(prediction)
        probabilities.append(score)
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} done; current task prediction: {prediction}, {text}")
    metrics = compute_metrics(labels, predictions, probabilities)
    print(metrics)


if __name__ == '__main__':
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    dataset = prepare_toxic_chat_test_data()
    openai_test(dataset=dataset)
    print(f"OpenAI Experiments Done.")
