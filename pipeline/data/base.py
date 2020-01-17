from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase


class DataBuilderBase(ABC):
    def __init__(self, config: ConfigBase):
        self.config = config

    @abstractmethod
    def build_training_data(self):
        pass

    @abstractmethod
    def build_inference_data(self):
        pass
