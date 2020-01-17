import abc
from pipeline.logging.logger import logger
import pandas as pd
from typing import List
from configuration import EVENT_COLUMNS


class FeatureExtractorBase(abc.ABC):
    @abc.abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def __repr__(self):
        return self.__class__.__name__


class FeatureExtractorCombiner(FeatureExtractorBase):
    def __init__(self,
                 feature_extractors: List[FeatureExtractorBase],
                 add_extractor_prefix_name: bool = False
                 ):
        self.add_extractor_prefix_name = add_extractor_prefix_name
        self._feature_extractors = feature_extractors

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("start extract features from combiner")

        result = data.copy(deep=True)
        logger.debug(f"have columns {data.columns}")

        candidates_columns_len = len(EVENT_COLUMNS)

        candidates = data[EVENT_COLUMNS]

        for feature_extractor in self._feature_extractors:
            logger.info(f"start extract from {repr(feature_extractor)}")

            features = feature_extractor.extract(data.copy())
            if features.shape[0] != data.shape:
                logger.warning(f"different shapes {features.shape[0]} and must be - {data.shape}, try drop duplicates")
                features = features.drop_duplicates(subset=[EVENT_COLUMNS])

                if features.shape[0] != data.shape:
                    logger.error(f"different shape after drop duplicates")

            features_count = len(features.columns) - candidates_columns_len

            logger.debug(f"get {features_count} features")
            logger.debug(f"feature columns = {features.columns}")

            if features_count == 0:
                logger.warning(f"{repr(feature_extractor)} doesnt return features")

            logger.debug(f"shape before merge {result.shape}")
            result = result.merge(features, on=EVENT_COLUMNS, how="left")
            logger.debug(f"shape after merge {result.shape}")

        return result

    def __repr__(self):
        reprs = [repr(feature_extractor) for feature_extractor in self._feature_extractors]
        return "combiner_{}_".format("_".join(reprs))
