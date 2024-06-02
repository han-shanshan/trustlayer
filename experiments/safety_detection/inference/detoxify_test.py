from transformers import pipeline

from experiments.safety_detection.inference.azure_test import prepare_toxic_chat_test_data
from training.training_engine import compute_metrics


def classify_records(dataset, pipe, threshold=0.5):
    labels = dataset["train"]["label"]
    probabilities = []
    predictions = []
    for text in dataset["train"]["text"]:
        result = pipe(text)
        print(f"result = {result}")
        score = 0
        for d in result[0]:
            if d['score'] > score:
                score = d['score']
        if score > threshold:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)
        probabilities.append(score)
    metrics = compute_metrics(labels, predictions, probabilities)
    print(metrics)


def detoxify_bert_test(data, threshold=0.5):
    detoxify_pipeline = pipeline(
        'text-classification',
        model='unitary/toxic-bert',
        tokenizer='bert-base-uncased',
        function_to_apply='sigmoid',
        return_all_scores=True
    )
    classify_records(dataset=data, pipe=detoxify_pipeline, threshold=threshold)


def detoxify_roberta_test(data, threshold=0.5):
    detoxify_pipeline = pipeline(
        'text-classification',
        model='unitary/unbiased-toxic-roberta',
        tokenizer='bert-base-uncased',
        function_to_apply='sigmoid',
        return_all_scores=True
    )
    classify_records(dataset=data, pipe=detoxify_pipeline, threshold=threshold)


if __name__ == '__main__':
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    dataset = prepare_toxic_chat_test_data()
    print(dataset)
    detoxify_bert_test(dataset)
    print(f"Detoxify Experiments Done.")
