from wrapper.wrapper_base import BaseWrapper
import re
import requests
from requests.exceptions import HTTPError, Timeout, RequestException


def extract_urls(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(pattern, text)
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
        web_response = requests.get(url, timeout=2)
        status_code_first_digit = int(str(web_response.status_code)[0])
        if status_code_first_digit == 2:
            print(
                f"Success! {url}, status: {web_response.status_code}, {general_statuses[status_code_first_digit]}: {statuses[web_response.status_code]}")
            return True
        else:
            print(
                f"Failure! {url}, status: {web_response.status_code}, {general_statuses[status_code_first_digit]}: {statuses[web_response.status_code]}")
            return False
    except HTTPError as http_err:
        status_code_first_digit = int(str(http_err.response.status_code)[0])
        print(
            f"Failure! {url}, status: {http_err.response.status_code}, {general_statuses.get(status_code_first_digit, 'Error')}: {statuses[http_err.response.status_code]}")
        return False
    except Timeout:
        print("Failure! Timeout error. The request timed out.")
        return False
    except RequestException as e:  # General catch-all for requests exceptions that are not Timeout or HTTPError
        print(f"Request failed: {e}")
        return False


class URLDetectionWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)

    def process(self, original_text):
        result = ""
        urls = extract_urls(original_text)
        if urls:
            for url in urls:
                result = result + f"Found URL: {url}. "
                if is_url_reachable(url):
                    result = result + f"The URL {url} is reachable. "
                    # if is_url_malicious(url): https://huggingface.co/DunnBC22/codebert-base-Malicious_URLs
                    #     result = result + f"Warning: The URL {url} is malicious."
                    # else:
                    #     result = result + f"The URL {url} is not malicious."
                else:
                    result = result + f"The URL {url} is not reachable."
        else:
            result = result + "No URLs found in the text. "
        return result
