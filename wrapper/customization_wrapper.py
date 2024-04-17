from wrapper.substitute_sensitive_info import SubstituteWrapper


class CustomizationWrapper:
    def __init__(self, config):
        self.wrappers = []
        if 'enable_substitution' in config and config['enable_substitution']:
            self.wrappers.append(("substitution", SubstituteWrapper(config)))

    def get_post_process_text(self, text: str):
        for wrapper_type, wrapper_entity in self.wrappers:
            text = wrapper_entity.process(text)
        return text
