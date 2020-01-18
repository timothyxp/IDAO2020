import pickle
import json
import numpy as np


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(path, obj, use_np_encoder: bool = True):
    encoder_class = None
    if use_np_encoder:
        encoder_class = NpEncoder

    with open(path, "w") as f:
        json.dump(obj, f, cls=encoder_class)


def load_json(path):
    with open(path, "r") as f:
        obj = json.load(f)

    return obj
