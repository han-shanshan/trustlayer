import sys
from customized_hallucination_pipeline.online_processing.post_processing import PostProcessor
from customized_hallucination_pipeline.online_processing.grounding_tool import HallucinationGroundingTool
from customized_hallucination_pipeline.test_use_case import warranty_use_case, fedml_security_detector_config
from safety_detection.safety_detector import SafetyDetector

use_case = warranty_use_case


def fake_GPT4_API_for_testing(text=None):
    return use_case["response"]


if __name__ == '__main__':
    prompt = use_case["prompt"]
    input_trust_detector = SafetyDetector(fedml_security_detector_config)

    if input_trust_detector.is_unsafe_content_detected(prompt):
        print("Unsafe content detected. Exiting program.")
        sys.exit(1)

    grounding_tool = HallucinationGroundingTool()
    enhanced_prompt = grounding_tool.grounding(prompt)

    to_ask = True
    while to_ask:
        raw_output = fake_GPT4_API_for_testing(enhanced_prompt)
        post_processor = PostProcessor()
        new_text, to_ask, logging = post_processor.post_processing(raw_output, reference_text=enhanced_prompt)

        if to_ask:
            enhanced_prompt = "The original query is" + enhanced_prompt + ", the output is: " + new_text + ", however, " \
                     "there are some errors with it, as follows: " + logging + ". Please fix the errors and " \
                                                                               "regenerate responses"
        else:
            print(f"The output is safe: {new_text}")
        print("===================")
