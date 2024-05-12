import logging
import os

from data_operation.data_reader import DataReader
from wrapper.wrapper_base import BaseWrapper
import re
from requests.exceptions import HTTPError, Timeout, RequestException
import requests
from pysafebrowsing import SafeBrowsing
"""Other methods to detect malicious URLs: https://huggingface.co/DunnBC22/codebert-base-Malicious_URLs """


def extract_urls(original_text):
    # pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    pattern = r'https?://(?:(?!https?://)[a-zA-Z0-9$-_@.&+!*\\(\\),%])+'

    urls = re.findall(pattern, original_text)
    return urls


def is_url_reachable(url):
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    general_statuses = {
        1: "Informational Responses",
        2: "Successful Responses",
        3: "Redirection Messages",
        4: "Client Error Responses",
        5: "Server Error Responses"
    }
    statuses = {
        100: "Continue",
        101: "Switching Protocols",
        102: "Processing (WebDAV)",
        103: "Early Hints",
        200: "OK",
        202: "Accepted",
        203: "Non-Authoritative Information",
        204: "No Content",
        205: "Reset Content",
        206: "Partial Content",
        207: "Multi-Status (WebDAV)",
        208: "Already Reported (WebDAV)",
        226: "IM Used (HTTP Delta encoding)",
        300: "Multiple Choices",
        301: "Moved Permanently",
        302: "Found",
        303: "See Other",
        304: "Not Modified",
        305: "Use Proxy Deprecated",
        306: "unused",
        307: "Temporary Redirect",
        308: "Permanent Redirect",
        400: "Bad Request",
        401: "Unauthorized",
        402: "Payment Required Experimental",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        407: "Proxy Authentication Required",
        408: "Request Timeout",
        409: "Conflict",
        410: "Gone",
        411: "Length Required",
        412: "Precondition Failed",
        413: "Payload Too Large",
        414: "URI Too Long",
        415: "Unsupported Media Type",
        416: "Range Not Satisfiable",
        417: "Expectation Failed",
        418: "I'm a teapot",
        421: "Misdirected Request",
        422: "Unprocessable Content (WebDAV)",
        423: "Locked (WebDAV)",
        424: "Failed Dependency (WebDAV)",
        425: "Too Early Experimental",
        426: "Upgrade Required",
        428: "Precondition Required",
        429: "Too Many Requests",
        431: "Request Header Fields Too Large",
        451: "Unavailable For Legal Reasons",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported",
        506: "Variant Also Negotiates",
        507: "Insufficient Storage (WebDAV)",
        508: "Loop Detected (WebDAV)",
        510: "Not Extended",
        511: "Network Authentication Require"
    }
    try:
        logging.info(f"url = {url}")
        web_response = requests.get(url, timeout=2)
        status_code_first_digit = int(str(web_response.status_code)[0])
        if status_code_first_digit == 2:
            logging.info(
                f"Success! {url}, status: {web_response.status_code}, {general_statuses[status_code_first_digit]}: {statuses[web_response.status_code]}")
            return True
        else:
            logging.info(
                f"Failure! {url}, status: {web_response.status_code}, {general_statuses[status_code_first_digit]}: {statuses[web_response.status_code]}")
            return False
    except HTTPError as http_err:
        status_code_first_digit = int(str(http_err.response.status_code)[0])
        logging.info(
            f"Failure! {url}, status: {http_err.response.status_code}, {general_statuses.get(status_code_first_digit, 'Error')}: {statuses[http_err.response.status_code]}")
        return False
    except Timeout:
        logging.info("Failure! Timeout error. The request timed out.")
        return False
    except RequestException as e:  # General catch-all for requests exceptions that are not Timeout or HTTPError
        logging.info(f"Request failed: {e}")
        return False


def is_phishing_url_with_phishtank(url):  # the API is not working currently
    endpoint = "https://checkurl.phishtank.com/checkurl/"
    response = requests.post(endpoint, data={"url": url, "format": "json"})
    logging.info(response)


class URLDetectionWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = DataReader.read_google_apikey()

    def is_phishing_url_with_google_safe_browsing(self, url):
        # e.g., http://malware.testing.google.test/testing/malware/', http://ianfette.org
        # This API & key can only be utilized on public data, e.g., public Maps data showing restaurant information.
        # Fail to detect some phishing urls
        s = SafeBrowsing(key=self.api_key)
        r = s.lookup_urls([url])
        # print(f"url = {url}, r = {r}")
        return r[url]['malicious']

    def get_malicious_urls(self, url_list):
        malicious_url_list = []
        s = SafeBrowsing(key=self.api_key)
        r = s.lookup_urls(url_list)
        # print(f"url = {url_list}, r = {r}")
        for k in r.keys():
            if r.get(k).get('malicious'):
                malicious_url_list.append(k)
        return malicious_url_list

    def process(self, original_text):
        urls = extract_urls(original_text)
        if not urls:
            return original_text
        malicious_urls = self.get_malicious_urls(urls)
        unreachable_urls = []
        for url in set(urls).difference(set(malicious_urls)):
            if not is_url_reachable(url):
                unreachable_urls.append(url)
        result = f"Detected URLs: {urls}. "
        if len(malicious_urls) > 0:
            result = f"{result}Malicious URLs: {malicious_urls}. "
        if len(unreachable_urls) > 0:
            result = f"{result}Unreachable URLs: {unreachable_urls}. "
        return f"[{result}] {original_text}"

    def process_problematic_urls(self, original_text):
        urls = extract_urls(original_text)
        if not urls:
            return original_text
        malicious_urls = self.get_malicious_urls(urls)
        unreachable_urls = []
        for url in set(urls).difference(set(malicious_urls)):
            if not is_url_reachable(url):
                unreachable_urls.append(url)
        result = f"Detected URLs: {urls}. "
        problematic_list = malicious_urls
        problematic_list.extend(x for x in unreachable_urls if x not in malicious_urls)
        if len(problematic_list) > 0:
            result = f"{result}Problematic URLs: {problematic_list}. "
        return f"[{result}] {original_text}"

