from datasets import load_dataset  # pip install google-api-python-client
from googleapiclient import discovery
import json
from data_operation.data_reader import DataReader
from training.training_engine import compute_metrics

GOOGLE_API_KEY = DataReader().read_google_apikey(is_perspective_api=True)


def perspectiveAPI_test(dataset, threshold=0.7):
    import ast
    import time
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=GOOGLE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    labels = dataset["train"]["label"]
    probabilities = []
    predictions = []
    counter = 0

    for text in dataset["train"]["text"]:
        counter += 1
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {},
                                    'PROFANITY': {}, 'THREAT': {}}
        }
        response = client.comments().analyze(body=analyze_request).execute()
        result = ast.literal_eval(json.dumps(response, indent=2))
        score = 0
        prediction_label = 0
        for item in ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']:
            if result['attributeScores'][item]['summaryScore']['value'] > score:
                score = result['attributeScores'][item]['summaryScore']['value']
        if score > threshold:
            prediction_label = 1
        probabilities.append(score)
        predictions.append(prediction_label)
        if counter == 55:
            time.sleep(60)
            counter = 0
        # print(f"{text}, \n prediction_label = {prediction_label}")
    metrics = compute_metrics(labels, predictions, probabilities)
    print(metrics)


if __name__ == '__main__':
    dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    print(dataset)
    perspectiveAPI_test(dataset=dataset, threshold=0.5)
