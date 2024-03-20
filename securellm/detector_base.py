from abc import ABC, abstractmethod
from typing import Union, Optional


class BaseSafetyDetector(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    def get_detection_results(self, text: Union[str, list], optional_text: Optional[Union[str, list]] = None):
        pass

