from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase
import pandas as pd
from catboost import CatBoostRegressor
import random
from pipeline.logging.logger import logger
import os
import pickle


class ModelBase(ABC):
    def __init__(self, config: ConfigBase):
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class ModelBoosting(ModelBase):
    def train(self):
        data = pd.read_csv(self.config.train_data_path, low_memory=False)

        list600 = [i for i in range(1, 601)]
        train_ids = random.choices(list600, k=500)
        test_ids = list(set(list600) ^ set(train_ids))

        target_columns = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
        drop_columns = ['id', 'epoch']
        cropped_data = data.drop(columns=(drop_columns + target_columns))

        train_data = cropped_data.loc[cropped_data['sat_id'].isin(train_ids)].drop(columns=['sat_id'])
        train_target = data[target_columns]

        model = CatBoostRegressor(
            learning_rate=0.07,
            max_depth=2,
            iterations=70,
            thread_count=8,
            loss_function='MultiRMSE'
        )

        logger.info("start a model fitting")

        model.fit(train_data, train_target)

        main_model_path = os.path.join(self.config.models_path, "main_model.pkl")

        logger.info(f"saving the model to {main_model_path}")
        with open(main_model_path, "wb") as f:
            pickle.dump(model, f)

        test_data = cropped_data.loc[cropped_data['sat_id'].isin(test_ids)].drop(columns=['sat_id'])
        test_data.to_csv(self.config.inference_data, index=False)

    def predict(self):
        logger.info("read inference_data")
        inference_data = pd.read_csv(self.config.inference_data, low_memory=False)

        logger.info(f"inference data shape {inference_data.shape}")

        main_model_path = os.path.join(self.config.models_path, "main_model.pkl")
        logger.info(f"loading model from {main_model_path}")
        with open(main_model_path, "rb") as f:
            model = pickle.loads(f.read())

        logger.info("start predicting")
        prediction = model.predict(inference_data)



        prediction.to_csv(self.config.submit_data_path, index=True)
