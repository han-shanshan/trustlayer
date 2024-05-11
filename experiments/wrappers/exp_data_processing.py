from datasets import load_dataset
import random

PROBLEMATIC_URLS = ["http://www.urltocheck1.org/", "http://www.urltocheck2.org/", "http://www.urltocheck3.com/",
                    "http://www.examplle.com", "http://www.serverdownexample.com"]
RANDOM_SEED = 0  # 42
MALICIOUS_PROB = 0.2
SELECTED_DATA_NUM_EACH_DATASET = 5
SELECTED_KEYS = ["title", "question", "answer", "description", "overview", "target"]


def select_random_data(dataset, n, problematic_percent):
    train_dataset = dataset['train']
    random.seed(RANDOM_SEED)
    selected_indices = random.sample(range(len(train_dataset)), n)
    selected_samples = [train_dataset[i] for i in selected_indices]
    processed_texts = []
    counter = 0
    for sample in selected_samples:
        text = ""
        for k in SELECTED_KEYS:
            if k in sample:
                text = text + sample[k]
        is_problematic_url_included = False
        if counter < int(problematic_percent * n):
            is_problematic_url_included = True
        counter = counter + 1
        processed_texts.append(insert_url_to_text(text, sample["url"], is_problematic_url_included).replace("\n", ""))
    return processed_texts


def url_exp_construct_data():
    dataset_faq = load_dataset("qgyd2021/e_commerce_customer_service", 'faq')
    dataset_product = load_dataset("qgyd2021/e_commerce_customer_service", 'product')
    dataset_reddit = load_dataset("marksverdhei/reddit-syac-urls")
    return select_random_data(dataset_faq, SELECTED_DATA_NUM_EACH_DATASET, MALICIOUS_PROB) \
           + select_random_data(dataset_product, SELECTED_DATA_NUM_EACH_DATASET, MALICIOUS_PROB) \
           + select_random_data(dataset_reddit, SELECTED_DATA_NUM_EACH_DATASET, MALICIOUS_PROB)


def insert_url_to_text(base_string, url, is_problematic_url_included):
    if not is_problematic_url_included:
        random_index = random.randint(0, len(base_string))
        new_string = base_string[:random_index] + url + base_string[random_index:]
        return new_string
    idx1, idx2 = random.sample(range(0, len(base_string)), 2)
    idx1, idx2 = sorted([idx1, idx2])

    problematic_url = PROBLEMATIC_URLS[random.randint(0, len(PROBLEMATIC_URLS))]
    new_string = base_string[:idx1] + url + base_string[idx1:idx2] + problematic_url + base_string[idx2:]
    return new_string


if __name__ == '__main__':
    url_exp_construct_data()
