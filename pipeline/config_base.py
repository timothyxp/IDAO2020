import os
from . import DATA_PATH
from pipeline.feature_extractors.base import FeatureExtractorBase
from pipeline.data.base import DataBuilderBase
from pipeline.models.base import ModelBase


class ConfigBase:
    def __init__(
            self,
            experiment_name: str,
            feature_extractor: FeatureExtractorBase = None,
            model: ModelBase = None,
            data_builder: DataBuilderBase = None
    ):
        self.experiment_name = experiment_name

        self.feature_extractor = feature_extractor

        self.data_path = os.path.join(DATA_PATH, self.experiment_name)

        os.makedirs(self.data_path, exist_ok=True)

        self.model = model

        self.data_builder = data_builder

    @property
    def inference_result_path(self):
        return f"{self.data_path}/inference_result.csv"

    @property
    def inference_data(self):
        return f"{self.data_path}/inference_data.csv"

    @property
    def train_data_path(self):
        return f"{self.data_path}/train_data.csv"

    @property
    def train_path(self):
        return f"{self.data_path}/train.csv"

    @property
    def test_path(self):
        return f"{self.data_path}/test.csv"

    @property
    def cache_path(self):
        return f"{self.data_path}/cache"
