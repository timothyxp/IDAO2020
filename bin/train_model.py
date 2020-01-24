from pipeline.utils import load_config
from pipeline.config_base import ConfigBase

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config: ConfigBase = load_config(args.config_path)
   
    config.model.train()


if __name__ == "__main__":
    main()
