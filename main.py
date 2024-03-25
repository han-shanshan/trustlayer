import sys
import yaml
from llm_inference.inference import Inference
from trust_libs.trust_detector import TrustDetector
from wrappers.trust_wrapper import TrustWrapper

if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file.read())  # how to use the config: e.g., config['feature_name']
    print(f"config: {config}")
    prompt = config['preprocessing']['prompt']
    print("---------------------Detect unsafe contents in input---------------------")
    input_trust_detector = TrustDetector(config['preprocessing_args'])

    if input_trust_detector.is_unsafe_content_detected(prompt, reference_text=None):
        print("Unsafe content detected. Exiting program.")
        sys.exit(1)  # Exit the entire script

    print("---------------------Inference---------------------")

    raw_output = Inference().infer(prompt)

    print("---------------------Detect unsafe contents in output---------------------")
    output_trust_detector = TrustDetector(config['postprocessing_args'])
    if output_trust_detector.is_unsafe_content_detected(raw_output, reference_text=prompt):
        print("Unsafe content detected. Exiting program.")
        sys.exit(1)  # Exit the entire script
    print("---------------------Post-process output---------------------")

    wrapper = TrustWrapper(config=config['postprocessing_args'])
    output = wrapper.get_post_process_text(raw_output)
    print(f"output = {output}")



    




