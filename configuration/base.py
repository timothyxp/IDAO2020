from pipeline.config_base import ConfigBase
from catboost import CatBoostRegressor
from pipeline.feature_extractors.base import FeatureExtractorCombiner, FeatureExtractorTimeStackWrapper
from pipeline.feature_extractors.time import FeatureExtractorTime
from pipeline.feature_extractors.radius import FeatureExtractorRadius
from pipeline.data.default import DataBuilderDefault
from pipeline.feature_extractors.base import FeatureExtractorCombiner
from pipeline.models.base import ModelBoosting
from sklearn.ensemble import RandomForestRegressor


class Config(ConfigBase):
    def __init__(self):
        feature_extractor = FeatureExtractorTimeStackWrapper(
            FeatureExtractorCombiner([
                FeatureExtractorRadius(),
                FeatureExtractorTime()
            ]),
            num_ex=5
        )

        model = ModelBoosting(self, lambda: CatBoostRegressor(
            learning_rate=0.1,
            max_depth=5,
            iterations=200,
            thread_count=8,
            loss_function='MAE'
        ))

        super().__init__(
            experiment_name="main",
            feature_extractor=feature_extractor,
            model=model,
            data_builder=DataBuilderDefault(self)
        )
