from pandas import datetime
from matplotlib import pyplot
from .base import ModelBase
from pipeline.logging.logger import logger
from configuration import TARGET_COLUMNS, EVENT_COLUMNS
from statsmodels.tsa.arima_model import ARIMA
from pipeline.helpers.dump_load import load_pickle, save_pickle
import random


class ModelARIMA(ModelBase):

    def train(self):
        logger.info("reading data")
        data = load_pickle(self.config.train_data_path)
        test_data = load_pickle(self.config.test_path)

        for metric in TARGET_COLUMNS:
            data[f"deviation_{metric}"] = data[metric] - data[f"{metric}_sim"]

        common_ids = list(set(data['sat_id'].tolist()) & set(test_data['sat_id'].tolist()))

        sim_cols = list(map(lambda s: s + '_sim', TARGET_COLUMNS))
        cropped_data = data.drop(columns=(sim_cols + TARGET_COLUMNS + ['id']))

        threshold = int(0.66 * cropped_data.shape[0])

        train_data = cropped_data.loc[cropped_data['sat_id'].isin(common_ids)][:threshold]
        train_target = data.loc[cropped_data['sat_id'].isin(common_ids)][TARGET_COLUMNS][:threshold]

        # сразу сделать необходимое число df group by sat_id

        for sat_id in common_ids:
            for metric in TARGET_COLUMNS:
                current_metric_data = pd.DataFrame(train_data[['deviation_' + metric, 'sat_id']])
                current_metric_data['epoch'] = train_data['epoch']  # pd.to_datetime(...)
                current_metric_data = current_metric_data.loc[current_metric_data['sat_id'] == sat_id].drop(
                    columns=['sat_id'])
                cur_met_nparr = current_metric_data.to_numpy()
                model = ARIMA(cur_met_nparr, order=(5, 1, 0)).fit(disp=0)
            break

    def predict(self):
        pass
