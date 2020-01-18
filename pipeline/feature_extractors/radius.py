from pipeline.feature_extractors.base import FeatureExtractorBase
import pandas as pd
import numpy as np


class FeatureExtractorRadius(FeatureExtractorBase):
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        data["r2"] = data["x_sim"] ** 2 + data["y_sim"] ** 2 + data["z_sim"] ** 2
        data["r"] = np.sqrt(data["r2"])

        data["x2"] = data["x_sim"] ** 2
        data["y2"] = data["y_sim"] ** 2
        data["z2"] = data["z_sim"] ** 2
        data["xy"] = np.sqrt(data["x2"] + data["y2"])
        data["xz"] = np.sqrt(data["x2"] + data["z2"])
        data["yz"] = np.sqrt(data["y2"] + data["z2"])

        data["Vr2"] = data["Vx_sim"] ** 2 + data["Vy_sim"] ** 2 + data["Vz_sim"] ** 2
        data["Vr"] = np.sqrt(data["Vr2"])

        data["Vx2"] = data["Vx_sim"] ** 2
        data["Vy2"] = data["Vy_sim"] ** 2
        data["Vz2"] = data["Vz_sim"] ** 2
        data["Vxy"] = np.sqrt(data["Vx2"] + data["Vy2"])
        data["Vxz"] = np.sqrt(data["Vx2"] + data["Vz2"])
        data["Vyz"] = np.sqrt(data["Vy2"] + data["Vz2"])

        return data
