from abc import ABC, abstractmethod
from pipeline.config_base import ConfigBase
import pandas as pd
from catboost import CatBoostRegressor
import random
from pipeline.logging.logger import logger
from pipeline.helpers.dump_load import load_pickle, save_pickle
import os
from configuration import TARGET_COLUMNS, EVENT_COLUMNS
from typing import Callable


class ModelBase(ABC):
    model: Callable[[], CatBoostRegressor]

    def __init__(self, config: ConfigBase, model):
        self.config = config
        self.model = model

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class ModelBoosting(ModelBase):
    def train(self):
        logger.info("reading data")
        data = load_pickle(self.config.train_data_path)

        list600 = list(range(1, 601))
        train_ids = random.choices(list600, k=500)
        test_ids = list(set(list600) ^ set(train_ids))

        drop_columns = ['id', 'epoch']
        cropped_data = data.drop(columns=(drop_columns + TARGET_COLUMNS))

        train_data = cropped_data.loc[cropped_data['sat_id'].isin(train_ids)].drop(columns=['sat_id'])
        train_target = data.loc[cropped_data['sat_id'].isin(train_ids)][TARGET_COLUMNS]

        test_data = cropped_data.loc[cropped_data['sat_id'].isin(test_ids)].drop(columns=['sat_id'])
        test_target = data.loc[cropped_data['sat_id'].isin(test_ids)][TARGET_COLUMNS]

        for metric in ["Vx", "Vy", "Vz", "x", "y", "z"]:
            model = self.model()

            logger.info(f"start a model fitting for metric = {metric}")

            metric_train_target = train_target[metric] - train_data[f"{metric}_sim"]
            metric_test_target = test_target[metric] - test_data[f"{metric}_sim"]

            model.fit(
                train_data,
                metric_train_target,
                use_best_model=True,
                eval_set=(test_data, metric_test_target),
                verbose=True
            )

            main_model_path = os.path.join(self.config.models_path, f"model__{metric}.pkl")

            logger.info(f"saving the model to {main_model_path}")
            save_pickle(main_model_path, model)

    def predict(self):
        logger.info("read inference_data")
        inference_data = load_pickle(self.config.inference_data)

        logger.info(f"inference data shape {inference_data.shape}")

        inference_for_predict = inference_data.drop(columns=EVENT_COLUMNS)

        for metric in ["Vx", "Vy", "Vz", "x", "y", "z"]:
            model_path = os.path.join(self.config.models_path, f"model__{metric}.pkl")

            logger.debug(f"loading model {model_path}")

            model = load_pickle(model_path)

            prediction = model.predict(inference_for_predict)

            prediction += inference_for_predict[f"{metric}_sim"]
            logger.info(f"prediction for {metric} finished")
            inference_data[metric] = prediction

        predictions = inference_data[["id"] + TARGET_COLUMNS]
        logger.info(f"predictions shape = {predictions.shape}")

        predictions.to_csv(self.config.submit_data_path, index=False)
