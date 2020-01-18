import pandas as pd
from datetime import datetime


# all common transformation here
def data_transformation(data: pd.DataFrame) -> pd.DataFrame:
    data.epoch = data.epoch.apply(lambda d: datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f'))

    return data
