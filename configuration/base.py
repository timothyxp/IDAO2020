from pipeline.config_base import ConfigBase
from catboost import CatBoostClassifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner, FeatureExtractorTimeStackWrapper
from pipeline.feature_extractors.time import FeatureExtractorTime
from pipeline.feature_extractors.radius import FeatureExtractorRadius
from pipeline.data.default import DataBuilderDefault
from pipeline.feature_extractors.base import FeatureExtractorCombiner
from pipeline.models.base import ModelBoosting


class Config(ConfigBase):
    def __init__(self):
        feature_extractor = FeatureExtractorTimeStackWrapper(
            FeatureExtractorCombiner([
                FeatureExtractorRadius(),
                FeatureExtractorTime()
            ])
        )

        model = ModelBoosting(self)

        super().__init__(
            experiment_name="main",
            feature_extractor=feature_extractor,
            model=model,
            data_builder=DataBuilderDefault(self)
        )
