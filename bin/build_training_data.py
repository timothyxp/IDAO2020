from pipeline.utils import load_config, run_train
from pipeline.config_base import ConfigBase

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config: ConfigBase = load_config(args.config_path)
    config.data_builder.build_training_data()


if __name__ == "__main__":
    main()