from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase
from pipeline.logging.logger import logger
import pandas as pd


class DataBuilderBase(ABC):
    def __init__(self, config: ConfigBase):
        self.config = config

    @abstractmethod
    def build_training_data(self):
        pass

    @abstractmethod
    def build_inference_data(self):
        logger.info("reading tables")
        test = pd.read_csv(self.config.test_path)

        logger.info("start extract features")
        features = self.config.feature_extractor.extract(test)

        logger.info(f"features shape {features.shape}")
        logger.info("saving data")
        features.to_csv(self.config.inference_data, index=False)
