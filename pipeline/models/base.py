from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase
import pandas as pd
from catboost import CatBoostRegressor
import random
from pipeline.logging.logger import logger
from pipeline.helpers.dump_load import load_pickle, save_pickle
import os
from configuration import TARGET_COLUMNS


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
        data = load_pickle(self.config.train_data_path)

        list600 = [i for i in range(1, 601)]
        train_ids = random.choices(list600, k=500)
        test_ids = list(set(list600) ^ set(train_ids))

        drop_columns = ['id', 'epoch']
        cropped_data = data.drop(columns=(drop_columns + TARGET_COLUMNS))

        train_data = cropped_data.loc[cropped_data['sat_id'].isin(train_ids)].drop(columns=['sat_id'])
        train_target = data.loc[cropped_data['sat_id'].isin(train_ids)][TARGET_COLUMNS]

        test_data = cropped_data.loc[cropped_data['sat_id'].isin(test_ids)].drop(columns=['sat_id'])
        test_target = data.loc[cropped_data['sat_id'].isin(test_ids)][TARGET_COLUMNS]

        model = CatBoostRegressor(
            learning_rate=0.07,
            max_depth=5,
            iterations=200,
            thread_count=8,
            loss_function='MultiRMSE'
        )

        logger.info("start a model fitting")

        model.fit(train_data, train_target, eval_set=(test_data, test_target), verbose=True)

        main_model_path = os.path.join(self.config.models_path, "main_model.pkl")

        logger.info(f"saving the model to {main_model_path}")
        save_pickle(main_model_path, model)

    def predict(self):
        logger.info("read inference_data")
        inference_data = load_pickle(self.config.inference_data)

        logger.info(f"inference data shape {inference_data.shape}")

        main_model_path = os.path.join(self.config.models_path, "main_model.pkl")
        logger.info(f"loading model from {main_model_path}")
        model = load_pickle(main_model_path)

        logger.info("start predicting")
        prediction = model.predict(inference_data)

        for i, column in enumerate(TARGET_COLUMNS):
            inference_data[column] = [k[i] for k in prediction]

        predictions = inference_data[["id"] + TARGET_COLUMNS]
        logger.info(f"predictions shape = {predictions.shape}")

        predictions.to_csv(self.config.submit_data_path, index=False)
