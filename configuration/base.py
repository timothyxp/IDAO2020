from pipeline.config_base import ConfigBase
from catboost import CatBoostClassifier
from pipeline.feature_extractors.base import FeatureExtractorCombiner


class Config(ConfigBase):
    def __init__(self):
        feature_extractor = FeatureExtractorCombiner([
        ])

        model = lambda: CatBoostClassifier(
            learning_rate=0.07,
            max_depth=2,
            iterations=70,
            thread_count=8
        )

        super().__init__(
            experiment_name="main",
            feature_extractor=feature_extractor,
            model=model
        )
