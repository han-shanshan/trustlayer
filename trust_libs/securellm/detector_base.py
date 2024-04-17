from abc import ABC, abstractmethod
from typing import Union, Optional


class BaseTrustDetector(ABC):
    def __init__(self, config):
        self.boundary = 0.5  # default 0.5; above 0.5 is detected as unsafe
        if config is not None and "boundary" in config and type(config['boundary']) is float and 0 < config['boundary'] < 1:
            self.boundary = config['boundary']

    def get_detection_results(self, text: Union[str, list], reference_text: Optional[Union[str, list]] = None):
        pass
