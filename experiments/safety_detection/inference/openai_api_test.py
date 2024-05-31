from datasets import load_dataset
from data_operation.data_reader import DataReader
from training.training_engine import compute_metrics
import requests


def openai_test(dataset):
    url = "https://api.openai.com/v1/moderations"
    key = DataReader().read_openai_apikey()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    labels = dataset["train"]["label"][:5]
    probabilities = []
    predictions = []
    for text in dataset["train"]["text"][:5]:
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
    metrics = compute_metrics(labels, predictions, probabilities)
    print(metrics)


if __name__ == '__main__':
    dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    print(dataset)
    openai_test(dataset=dataset)
