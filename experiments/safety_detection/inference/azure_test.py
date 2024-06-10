from data_operation.data_reader import DataReader
from experiments.safety_detection.inference.detoxify_test import prepare_toxic_chat_test_data
from training.training_engine import compute_metrics
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
# pip install azure-ai-contentsafety


def azure_test(dataset):
    import time
    key = DataReader().read_azure_apikey()
    end_point = "https://test-for-trustlayer.cognitiveservices.azure.com/"

    client = ContentSafetyClient(end_point, AzureKeyCredential(key))  # Create an Azure AI Content Safety client

    labels = dataset["label"]
    probabilities = []
    predictions = []
    counter = 0
    for text in dataset["text"]:
        request = AnalyzeTextOptions(text=text)
        try:
            response = client.analyze_text(request)
        except HttpResponseError as e:
            print("Analyze text failed.")
            if e.error:
                print(f"Error code: {e.error.code}")
                print(f"Error message: {e.error.message}")
                raise
            print(e)
            raise ValueError("Error")

        hate_result = next(item for item in response.categories_analysis if item.category == TextCategory.HATE)
        self_harm_result = next(
            item for item in response.categories_analysis if item.category == TextCategory.SELF_HARM)
        sexual_result = next(item for item in response.categories_analysis if item.category == TextCategory.SEXUAL)
        violence_result = next(item for item in response.categories_analysis if item.category == TextCategory.VIOLENCE)
        unsafe_severity = 0
        if hate_result and hate_result.severity > unsafe_severity:
            unsafe_severity = hate_result.severity
        if self_harm_result and self_harm_result.severity > unsafe_severity:
            unsafe_severity = self_harm_result.severity
        if sexual_result and sexual_result.severity > unsafe_severity:
            unsafe_severity = sexual_result.severity
        if violence_result and violence_result.severity > unsafe_severity:
            unsafe_severity = violence_result.severity
        prediction = 0
        if unsafe_severity > 0:
            prediction = 1
        predictions.append(prediction)
        probabilities.append(0)  # fake
        print(f"{counter}---{unsafe_severity}, {text} \n =========================\n")
        print(f"{counter}---{unsafe_severity}, {text} \n =========================\n")
        counter += 1
        if counter % 10 == 0:
            print(f"{counter} done; current task prediction: {prediction}, {text}")
            # time.sleep(10)

    metrics = compute_metrics(labels, predictions, probabilities)
    metrics.pop('roc_auc')  # probabilities are fake; just to make the parameters suitable for compute_metrics
    print(metrics)


if __name__ == '__main__':
    # dataset = load_dataset('csv', data_files="test_data/all_in_one_unsafe_contents_test_data.csv")
    dataset = prepare_toxic_chat_test_data()
    azure_test(dataset)
    print(f"Azure Experiments Done.")
