from wrapper.wrapper_base import BaseWrapper


class SuggestedReadingTimeWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)

        # The average reader can read 238 words per minute (WPM) while reading silently.
        # https://scholarwithin.com/average-reading-speed#:~:text=The%20average%20reader%20can%20read,of%20300%20words%20per%20minute.
        if 'avg_reading_words_per_min' in config and isinstance(config['avg_reading_words_per_min'], int) and config[
            'avg_reading_words_per_min'] > 0:
            self.avg_reading_words_per_min = config['avg_reading_words_per_min']
        else:
            self.avg_reading_words_per_min = 238

    def process(self, original_text):
        word_num = len(original_text.split())
        if word_num < self.avg_reading_words_per_min:
            reading_time = "{:.1f} seconds".format(word_num * 60 / self.avg_reading_words_per_min)
        else:
            reading_time = "{:.1f} minutes".format(word_num / self.avg_reading_words_per_min)
        return "[Suggested Reading Time: " + reading_time + "] " + original_text


if __name__ == '__main__':
    wrapper = SuggestedReadingTimeWrapper(config=None)
    print(f"dictionary = {wrapper.substitution_dictionary}")
    new_text = wrapper.process(
        "Despite being addicted to online shopping, she always hunted for cheap deals to mitigate her poor financial situation.")
    print(f"new text = {new_text}")
