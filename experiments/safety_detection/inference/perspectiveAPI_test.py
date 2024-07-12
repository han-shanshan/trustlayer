# pip install google-api-python-client
from googleapiclient import discovery
import json
from googleapiclient.errors import HttpError
from data_operation.data_file_operator import FileOperator
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from training.classification_training_engine import compute_metrics

GOOGLE_API_KEY = FileOperator().read_google_apikey(is_perspective_api=True)

"""
https://developers.perspectiveapi.com/s/about-the-api-score?language=en_US 
For social science researchers using Perspective to study harassment, we recommend experimenting with thresholds 
of 0.7 or 0.9, similar to typical moderation use cases. By comparison, machine learning researchers using Perspective 
to filter potentially toxic content from their data should use higher thresholds, like 0.9 or 0.95, as low thresholds 
increase the potential for bias """
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

    labels = dataset["label"]
    probabilities = []
    predictions = []
    counter = 0

    for text in dataset["text"]:
        counter += 1
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {},
                                    'PROFANITY': {}, 'THREAT': {}}
        }
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            result = ast.literal_eval(json.dumps(response, indent=2))
            score = 0
            prediction_label = 0
            for item in ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']:
                if result['attributeScores'][item]['summaryScore']['value'] > score:
                    score = result['attributeScores'][item]['summaryScore']['value']
            if score > threshold:
                prediction_label = 1
        except HttpError as error:
            error_details = json.loads(error.content.decode())
            print(f"Error occurred: text = {text}, error = {error_details}")

            if error.resp.status == 400 and 'LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE' in error.content.decode():
                print("Unsupported language for the requested attributes.")
                # the API fails to detect gibberish data
                prediction_label = 0
                score = 0
            else:
                # For other HTTP errors, you may want to re-raise them or handle differently
                raise ValueError(error_details)

        probabilities.append(score)
        predictions.append(prediction_label)
        if counter == 55:
            time.sleep(60)
            counter = 0
            print(f"{text}, \n prediction_label = {prediction_label}")
    metrics = compute_metrics(labels, predictions, probabilities)
    print(metrics)


if __name__ == '__main__':
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    dataset = prepare_toxic_chat_test_data()
    perspectiveAPI_test(dataset=dataset, threshold=0.5)
    print(f"PerspectiveAPI Experiments Done.")
