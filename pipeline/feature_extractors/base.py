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
                 ):
        self._feature_extractors = feature_extractors

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("start extract features from combiner")

        result = data.copy(deep=True)
        logger.debug(f"have columns {data.columns}")

        candidates_columns_len = len(EVENT_COLUMNS)

        for feature_extractor in self._feature_extractors:
            logger.info(f"start extract from {repr(feature_extractor)}")

            features = feature_extractor.extract(data.copy())
            if features.shape[0] != data.shape[0]:
                logger.warning(f"different shapes {features.shape[0]} and must be - {data.shape[0]}, try drop duplicates")
                features = features.drop_duplicates(subset=EVENT_COLUMNS)

                if features.shape[0] != data.shape[0]:
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


class FeatureExtractorTimeStackWrapper(FeatureExtractorBase):
    def __init__(self, feature_extractor: FeatureExtractorBase, num_ex = 5):
        self.feature_extractor = feature_extractor
        self.num_ex = num_ex

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        initial_columns = data.columns
        features = self.feature_extractor.extract(data)
        features_columns = list(set(features.columns).difference(initial_columns))
        logger.debug(f"have new features = {features_columns}")

        time_features = []
        for sat_id, group in features.groupby("sat_id"):
            for i in range(1, self.num_ex + 1):
                logger.debug(group.columns)

                shifted_df = group[features_columns].copy()
                shifted_df.index = shifted_df.index + i
                rename_map = dict()
                for column in features_columns:
                    rename_map[column] = f"{column}_shift_{i}"
                shifted_df = shifted_df.rename(columns=rename_map)

                group = group.merge(shifted_df, how="left", left_index=True, right_index=True)

            group["sat_id"] = sat_id
            time_features.append(group)

        time_features = pd.concat(time_features)

        return time_features
