############ TASK #############
SEMANTIC_TASK_NAME = "semantic"
TOPIC_TASK_NAME = "topic"
GIBBERISH_TASK_NAME = "gibberish"
UNSAFE_PROMPT_TASK_NAME = "unsafe_prompt"
HALLUCINATION_TASK_NAME = "hallucination"
HALLUCINATION_EXPLANATION_TASK_NAME = "hallucination_explanation"
TOXICITY_TASK_NAME = "toxicity"
ALL_IN_ONE_UNSAFE_CONTENTS_TASK_NAME = "all_in_one_unsafe_contents"
CUSTOMIZED_HALLUCINATION_TASK_NAME = "customized_hallucination"

############ MODEL #############
MODEL_NAME_BERT_BASE = "google-bert/bert-base-uncased"
MODEL_NAME_TINYLAMMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# FOX_BASE_GPU = "/raid/user/models/femma_1b_stage2-pt9/checkpoint-363750"
FOX_BASE_GPU = "/raid/user/models/femma_1b_stage3_aggressive_v2/checkpoint-23000"

EXPLANATION_RESPONSE_TEMPLATE = "### Response: "