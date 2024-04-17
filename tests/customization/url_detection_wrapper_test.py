import unittest
from unittest.mock import patch
from wrapper.url_detection_wrapper import extract_urls, URLDetectionWrapper


class TestURLDetectionWrapper(unittest.TestCase):

    def test_process_with_urls_found(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_urls = "Check out this link: https://www.google.com/"
        expected_result = "Found URL: https://www.google.com/. The URL https://www.google.com/ is reachable. "
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 200
            result = self.wrapper.process(text_with_urls)
            self.assertIn(expected_result, result, "The process method should detect and confirm reachable URLs.")

    def test_process_with_101_url(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_unreachable_url = "This site is down: http://testtesttest.com"
        expected_result = "Found URL: http://testtesttest.com. The URL http://testtesttest.com is not reachable."
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 101
            result = self.wrapper.process(text_with_unreachable_url)
            self.assertIn(expected_result, result, "The process method should detect unreachable URLs.")

    def test_process_with_404_url(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_unreachable_url = "This site is down: https://dfghjfghj.com"
        expected_result = "Found URL: https://dfghjfghj.com. The URL https://dfghjfghj.com is not reachable."
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 404
            result = self.wrapper.process(text_with_unreachable_url)
            self.assertIn(expected_result, result, "The process method should detect unreachable URLs.")

    def test_process_with_301_url(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_unreachable_url = "This site is down: http://testtesttest.com"
        expected_result = "Found URL: http://testtesttest.com. The URL http://testtesttest.com is not reachable."
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 301
            result = self.wrapper.process(text_with_unreachable_url)
            self.assertIn(expected_result, result, "The process method should detect unreachable URLs.")

    def test_process_with_401_url(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_unreachable_url = "This site is down: http://testtesttest.com"
        expected_result = "Found URL: http://testtesttest.com. The URL http://testtesttest.com is not reachable."
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 401
            result = self.wrapper.process(text_with_unreachable_url)
            self.assertIn(expected_result, result, "The process method should detect unreachable URLs.")

    def test_process_with_504_url(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_with_unreachable_url = "This site is down: http://testtesttest.com"
        expected_result = "Found URL: http://testtesttest.com. The URL http://testtesttest.com is not reachable."
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.status_code = 504
            result = self.wrapper.process(text_with_unreachable_url)
            self.assertIn(expected_result, result, "The process method should detect unreachable URLs.")

    def test_process_no_urls(self):
        self.wrapper = URLDetectionWrapper(config={})
        text_without_urls = "This text has no links."
        expected_result = "No URLs found in the text. "
        result = self.wrapper.process(text_without_urls)
        self.assertEqual(expected_result, result, "The process method should handle texts without URLs.")

    def test_extract_urls_function(self):
        self.wrapper = URLDetectionWrapper(config={})
        text = "Visit https://rtyuighjk.com and http://dfghjkcvbnm.com."
        urls = extract_urls(text)
        self.assertEqual(len(urls), 2, "extract_urls should find two URLs in the text.")


if __name__ == '__main__':
    unittest.main()
