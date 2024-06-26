############ TASK #############
SEMANTIC_TASK_NAME = "semantic"
TOPIC_TASK_NAME = "topic"
GIBBERISH_TASK_NAME = "gibberish"
UNSAFE_PROMPT_TASK_NAME = "unsafe_prompt"
HALLUCINATION_TASK_NAME = "hallucination"
HALLUCINATION_EXPLANATION_TASK_NAME = "hallucination_explanation"
TOXICITY_TASK_NAME = "toxicity"
ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME = "all_in_one_unsafe_contents"

############ MODEL #############
MODEL_NAME_BERT_BASE = "google-bert/bert-base-uncased"
MODEL_NAME_TINYLAMMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FOX = "tensoropera/Fox-1-1.6B"
# FOX_BASE_GPU = "/raid/user/models/femma_1b_stage3_aggressive_v2/checkpoint-23000"

EXPLANATION_RESPONSE_TEMPLATE = "### Response:"

###############
SINGLE_LABEL_CLASSIFICATION_PROBLEM_TYPE = "single_label_classification"
MULTI_LABEL_CLASSIFICATION_PROBLEM_TYPE = "multi_label_classification"
HALLUCINATION_REASONING_PROBLEM_TYPE = "hallucination_reasoning"