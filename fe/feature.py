import os
import pickle
import logging
import easydict
import os.path as p
from typing import List

import pandas as pd

from logger import get_logger

logger = get_logger("feature")


class FEBase:
    name: str = None  # Fature Engineering 이름
    pre_fe: set = None  # 선행되어야 하는 Feature Engineering
    no_fe: set = None  # 같이 사용되면 안되는 Feature Engineering

    @classmethod
    def get_save_path(cls):
        save_dir = p.join(os.environ["HOME"], "features")

        if not p.exists(save_dir):
            os.mkdir(save_dir)

        save_path = p.join(save_dir, f"{cls.name}.pkl")
        return save_path

    @classmethod
    def save_feature_df(cls, df: pd.DataFrame):
        save_path = cls.get_save_path()

        with open(save_path, "wb") as f:
            pickle.dump(df, f)

        print(f"save features dataframe to {save_path} ...")

    @classmethod
    def load_feature_df(cls):
        load_path = cls.get_save_path()

        with open(load_path, "rb") as f:
            right_df = pickle.load(f)

        print(f"load features {load_path} to dataframe ... ")
        return right_df


class FEPipeline:
    def __init__(self, args: easydict.EasyDict, fes: List[FEBase]):
        self.args = args
        self.fes = fes
        self.df = None

        assert "root_dir" in self.args, "args.root_dir을 설정해주세요."

        log_file_handler = logging.FileHandler(p.join(self.args.root_dir, "features.log"))
        logger.addHandler(log_file_handler)

    def description(self):
        logging.info("Feature Description")

        for fe in self.fes:
            logger.info(f"Feature Engineering Name: {fe.name}")
            for k, v in fe.description.items():
                logger.info(f"{k:<15} : {v}")

    def debug(self):
        pre_fe = set()

        for fe in self.fes:
            if fe.no_fe is not None:
                if len(fe.no_fe.intersection(pre_fe)) != 0:
                    raise ValueError(f"{fe.name}'s fe.no_fe: {fe.no_fe} in pre_fe({pre_fe})")
            if fe.pre_fe is not None:
                if len(fe.pre_fe.difference(pre_fe)) != 0:
                    raise ValueError(f"{fe.name}'s fe.pre_fe: {fe.pre_fe} not in pre_fe({pre_fe})")

            pre_fe.add(fe.name)

        logging.info("Debug Completed! You can transform!")

    def transform(self, df):
        logger.info("Feature Engineering Start ... ")
        original_columns = df.columns

        for fe in self.fes:
            df = fe.transform(df, self.args)
            logger.info(f"Feature Engineering Name: {fe.name}")

            for k, v in fe.description:
                logger.info(f"{k:<15} : {v}")
                logger.info("[Examples]")

                for idx in range(0, min(1000, len(df)), 100):
                    logger.info(f"INDEX {idx:<03}: {df.iloc[idx][k]}")

        logger.info("Feature Engineering End ... ")
        logger.info(f"Original DataFrame Keywords: {original_columns}")
        logger.info(f"Feature Added DataFrame Keywords: {df.columns}")

        self.df = df

    def get_feature_df(self, keys):
        return self.df[keys]
