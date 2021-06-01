import os
import pickle
import logging
import easydict
import os.path as p
from typing import List

import pandas as pd


logger = logging.getLogger("feature")
logger.setLevel(logging.INFO)


class FEBase:
    name: str = "base_feature"  # Fature Engineering 이름
    fe_type: str = "seq"
    pre_fe: set = None  # 선행되어야 하는 Feature Engineering
    no_fe: set = None  # 같이 사용되면 안되는 Feature Engineering
    description = {
            "userID": "사용자의 고유 번호입니다. 총 7,442명의 학생이 있습니다",
            "assessmentItemID": "사용자가 푼 문항의 일련 번호입니다.",
            "testID": "사용자가 푼 문항이 포함된 시험지의 일련 번호입니다.",
            "answerCode": "사용자가 푼 문항의 정답 여부를 담고 있는 이진 (0/1) 데이터입니다.",
            "Timestamp": "사용자가 문항을 푼 시간 정보입니다.",
            "KnowledgeTag": "사용자가 푼 문항의 고유 태그가 담겨져 있습니다.",
    }

    @classmethod
    def get_save_path(cls, is_train):
        save_dir = p.join(os.environ["HOME"], "features")
        prefix = "train" if is_train else "test"

        if not p.exists(save_dir):
            os.mkdir(save_dir)

        save_path = p.join(save_dir, f"{prefix}_{cls.name}.pkl")
        return save_path

    @classmethod
    def save_feature_df(cls, df: pd.DataFrame, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(df, f)

        print(f"save features dataframe to {save_path} ...")

    @classmethod
    def load_feature_df(cls, load_path):
        with open(load_path, "rb") as f:
            right_df = pickle.load(f)

        print(f"load features {load_path} to dataframe ... ")
        return right_df

    @classmethod
    def transform(cls, df):
        raise NotImplementedError


class FEPipeline:
    def __init__(self, args: easydict.EasyDict, fes: List[FEBase]):
        self.args = args
        self.fes = fes
        self.df = None

        assert "root_dir" in self.args, "args.root_dir을 설정해주세요."

        log_file_handler = logging.FileHandler(p.join(self.args.root_dir, "features.log"))
        logger.addHandler(log_file_handler)

    def description(self):
        print("[Feature Descriptions]")

        for fe in [FEBase] + self.fes:
            print(f"\nfeature name : {fe.name}")
            print(f"feature type : {fe.fe_type}")
            for k, v in fe.description.items():
                print(f" - {k:<20} : {v}")

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

    def transform(self, df, is_train):
        logger.info("Feature Engineering Start ... ")
        original_columns = df.columns
        self.df = df

        for fe in self.fes:
            self.df = fe.transform(self.df, is_train)
            logger.info(f"\nFeature Engineering Name: {fe.name}")

            for k, v in fe.description.items():
                logger.info(f"\n{k:<15} : {v}")
                logger.info(f"dtype: {self.df[k].dtype}")
                logger.info("[Examples]")

                for idx in range(0, min(1000, len(self.df)), 100):
                    logger.info(f"INDEX {idx:<04}: {self.df.iloc[idx][k]}")

        logger.info("Feature Engineering End ... ")
        logger.info(f"Original DataFrame Keywords: {original_columns}")
        logger.info(f"Feature Added DataFrame Keywords: {self.df.columns}")

        return self.df

    def get_feature_df(self, df, keys):
        return df[keys]
