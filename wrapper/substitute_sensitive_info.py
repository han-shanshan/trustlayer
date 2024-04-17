from storage.basic_data_storage_operations import read_substitutions_from_file
from wrapper.wrapper_base import BaseWrapper


class SubstituteWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        if "substitution_dictionary=None" in config:
            self.substitution_dictionary = config['substitution_dictionary']
        else:
            self.substitution_dictionary = read_substitutions_from_file("./storage/substitutions.txt")

    def process(self, original_text):
        for k in self.substitution_dictionary.keys():
            lower_k = k.lower()
            first_capital_k = self.set_first_letter_to_capital(lower_k)
            if lower_k in original_text:
                original_text = original_text.replace(lower_k, self.substitution_dictionary.get(k))
            if first_capital_k in original_text:
                original_text = original_text.replace(first_capital_k, self.set_first_letter_to_capital(self.substitution_dictionary.get(k)))
        return original_text

    @staticmethod
    def set_first_letter_to_capital(original_word):
        new_word = ""
        if len(original_word) > 1:
            new_word = original_word[0].upper() + original_word[1:]
        if len(original_word) == 1:
            new_word = original_word[0].upper()
        return new_word


if __name__ == '__main__':
    wrapper = SubstituteWrapper(config=None)
    print(f"dictionary = {wrapper.substitution_dictionary}")
    new_text = wrapper.process("Despite being addicted to online shopping, she always hunted for cheap deals to mitigate her poor financial situation.")
    print(f"new text = {new_text}")

