from pipeline.feature_extractors.base import FeatureExtractorBase
import pandas as pd


class FeatureExtractorTime(FeatureExtractorBase):
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        data["day"] = [d.day for d in data["epoch"]]
        data["hour"] = [d.hour for d in data["epoch"]]
        data["minute"] = [d.minute for d in data["epoch"]]
        data["second"] = [d.second for d in data["epoch"]]

        return data
