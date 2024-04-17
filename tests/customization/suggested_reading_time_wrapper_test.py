import unittest
from wrapper.suggested_reading_time_wrapper import SuggestedReadingTimeWrapper


class TestSuggestedReadingTimeWrapper(unittest.TestCase):

    def test_short_text_reading_time(self):
        config = {}
        wrapper = SuggestedReadingTimeWrapper(config)
        result = wrapper.process("This is a short text.")
        print(f"test_short_text_reading_time: {result}")
        self.assertIn("seconds", result, "Reading time should be in seconds for short text.")

    def test_long_text_reading_time(self):
        config = {}
        wrapper = SuggestedReadingTimeWrapper(config)
        result = wrapper.process("This is a long text. " * 100)  # Creates a text longer than 238 words
        print(f"test_short_text_reading_time: {result}")
        self.assertIn("minutes", result, "Reading time should be in minutes for long text.")

    def test_default_avg_reading_wpm(self):
        config = {}
        wrapper = SuggestedReadingTimeWrapper(config)
        self.assertEqual(wrapper.avg_reading_words_per_min, 238, "Default avg_reading_words_per_min should be 238.")

    def test_custom_avg_reading_wpm(self):
        config = {'avg_reading_words_per_min': 300}
        wrapper = SuggestedReadingTimeWrapper(config)
        self.assertEqual(wrapper.avg_reading_words_per_min, 300, "Custom avg_reading_words_per_min should be set "
                                                                 "correctly.")

    def test_empty_text(self):
        config = {}
        wrapper = SuggestedReadingTimeWrapper(config)
        result = wrapper.process("")
        self.assertIn("0 seconds", result, "Empty text should have a reading time of 0 seconds.")


if __name__ == '__main__':
    unittest.main()
