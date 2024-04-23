import sys
from hallucination_pipeline.post_processing import PostProcessor
from hallucination_pipeline.grounding_tool import GroundingTool
from hallucination_pipeline.test_use_case import warranty_use_case
from trust_libs.trust_detector import TrustDetector

use_case = warranty_use_case


def fake_llm_for_testing(text=None):
    return use_case["response"]


if __name__ == '__main__':
    prompt = use_case["prompt"]
    fedml_security_detector_config = {
        "enable_toxicity": True,
        "enable_gibberish": True,
        "enable_unsafe_prompt": True
    }
    input_trust_detector = TrustDetector(fedml_security_detector_config)

    if input_trust_detector.is_unsafe_content_detected(prompt):
        print("Unsafe content detected. Exiting program.")
        sys.exit(1)

    grounding_tool = GroundingTool()
    enhanced_prompt = grounding_tool.grounding(prompt)
    print(enhanced_prompt)

    to_ask = True
    while to_ask:
        raw_output = fake_llm_for_testing(enhanced_prompt)
        post_processor = PostProcessor()
        new_text, to_ask, logging = post_processor.post_processing(raw_output)

        if to_ask:
            enhanced_prompt = "The original query is" + enhanced_prompt + ", the output is: " + new_text + ", however, " \
                     "there are some errors with it, as follows: " + logging + ". Please fix the errors and " \
                                                                               "regenerate responses"
