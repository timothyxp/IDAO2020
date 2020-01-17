from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase


class ModelBase(ABC):
    def __init__(self, config: ConfigBase):
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
