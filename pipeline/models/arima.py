import os
import pandas as pd
from .base import ModelBase
from pipeline.logging.logger import logger
from configuration import TARGET_COLUMNS, EVENT_COLUMNS
from statsmodels.tsa.arima_model import ARIMA
from pipeline.helpers.dump_load import load_pickle, save_pickle


class ModelARIMA(ModelBase):

    def train(self):
        logger.info("reading data")
        data = load_pickle(self.config.train_data_path)
        test_data = load_pickle(self.config.test_path)

        for metric in TARGET_COLUMNS:
            data[f"deviation_{metric}"] = data[metric] - data[f"{metric}_sim"]

        common_ids = list(set(data['sat_id'].tolist()) & set(test_data['sat_id'].tolist()))

        sim_cols = list(map(lambda s: s + '_sim', TARGET_COLUMNS))
        cropped_data = data.drop(columns=(sim_cols + ['id']))
        cropped_data = cropped_data.loc[cropped_data['sat_id'].isin(common_ids)]

        threshold = int(0.66 * cropped_data.shape[0])

        train_data = cropped_data.drop(columns=TARGET_COLUMNS)[:threshold]
        train_sats = [x for x in train_data.groupby('sat_id')]  # returns tuple (id, df)

        # train_target = cropped_data[TARGET_COLUMNS][:threshold]

        models_by_sat = []

        for sat_data in train_sats:
            models_by_sat.append([])
            for metric in TARGET_COLUMNS:
                current_metric_data = pd.DataFrame(sat_data[1]['deviation_' + metric])
                current_metric_data['epoch'] = pd.to_datetime(sat_data[1]['epoch'])
                current_metric_data.set_index('epoch', inplace=True)

                logger.info(f"start a model fitting for metric={metric} and id={sat_data[0]}")

                models_by_sat[-1].append((sat_data[0], ARIMA(current_metric_data, order=(1, 1, 1)).fit(disp=0)))

        model_path = os.path.join(self.config.models_path, f"arima__{metric}.pkl")
        logger.info(f"saving the model to {model_path}")
        save_pickle(model_path, models_by_sat)


    def predict(self):
        pass
