from transformers import pipeline
import os
from data_operation.data_loader import DataLoader
from training.training_engine import compute_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def prepare_toxic_chat_test_data():
    toxic_chat_data = DataLoader().process_a_subdataset_for_all_in_one_task(dataset_type="toxic-chat")
    # use English && human annotation for testing
    print(toxic_chat_data)
    return toxic_chat_data

def classify_records(dataset, pipe, threshold=0.5):
    labels = dataset["label"]
    probabilities = []
    predictions = []
    counter = 0
    for text in dataset["text"]:
        result = pipe(text, padding=True, truncation=True)
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
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} done. score = {score}, prediction = {prediction}, text = {text}")
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
    detoxify_roberta_test(dataset)
    print(f"Detoxify Experiments Done.")
