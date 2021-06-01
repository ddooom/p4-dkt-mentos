import logging
import os.path as p

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from logger import get_logger

logger = get_logger("preprocess")
logger.setLevel(logging.INFO)


class Preprocess:
    def __init__(self, args, fe_pipeline, columns):
        self.args = args
        self.datas = {"train": None, "valid": None, "test": None}
        self.fe_pipeline = fe_pipeline
        self.columns = columns

        assert args.data_dir, f"{args.data_dir}를 설정해주세요."
        self.datas["train"] = pd.read_csv(p.join(args.data_dir, "train_data.csv"))
        self.datas["test"] = pd.read_csv(p.join(args.data_dir, "test_data.csv"))

        self.datas["org_train"] = self.datas["train"].copy()
        self.datas["org_test"] = self.datas["test"].copy()

        log_file_handler = logging.FileHandler(p.join(self.args.root_dir, "preprocess.log"))
        logger.addHandler(log_file_handler)
        logger.addHandler(logging.StreamHandler())

    def get_data(self, key="train"):
        """ return datas """
        assert key in self.datas.keys(), f"{key} not in {self.datas.keys()}"
        assert self.datas[key] is not None, f"{key} dataset is None"
        return self.datas[key]

    def split_data(self, seed=42):
        """ User 기준으로 Split 한다. """
        logger.info("Split based on User")
        logger.info(f"Original Train Dataset: {len(self.datas['train'])}")

        user_id = self.datas["train"].userID.unique()
        t_idx, v_idx = train_test_split(user_id, test_size=0.1, random_state=seed)
        t_idx = set(t_idx)

        fancy_index = self.datas["train"]["userID"].apply(lambda x: x in t_idx)
        train = self.datas["train"][fancy_index]
        valid = self.datas["train"][(fancy_index - 1).astype(bool)]

        self.datas["train"] = train
        self.datas["valid"] = valid

        logger.info(f"Split Train Dataset: {len(self.datas['train'])}")
        logger.info(f"Split Valid Dataset: {len(self.datas['valid'])}")

    def _make_rows_to_sequence(self, r):
        datas = []
        for key in self.columns:
            if key == "userID":
                continue
            datas.append(r[key].values)
        return tuple(datas)

    def _data_augmentation_test_dataset(self):
        """ 1 테스트 데이터셋 추가 """

        logger.info("Use the test datast for data augmentation")
        logger.info(f"Before Length: {len(self.datas['train'])}")

        test_dataset = self.datas["test"]
        aug_test_dataset = test_dataset[test_dataset["userID"] == test_dataset["userID"].shift(-1)]
        self.datas["train"] = pd.concat([self.datas["train"], aug_test_dataset], axis=0)
        logger.info(f"After Length: {len(self.datas['train'])}")

    def _data_augmentation_user_testpaper(self):
        """ 2 사이클 증강 """

        logger.info("Group By (userID, testPaper)")
        grouped = self.datas["train"].groupby(["userID", "testPaper"]).apply(lambda r: self._make_rows_to_sequence(r))
        self.datas["train_grouped"] = grouped.values
        logger.info(f"Group By (userID, testPaper) Length: {len(self.datas['train_grouped'])}")

    def _data_augmentation_user_testid(self):
        """ 3 대분류 증강 """

        logger.info("Group By (userID, firstClass)")
        grouped = self.datas["train"].groupby(["userID", "firstClass"]).apply(lambda r: self._make_rows_to_sequence(r))
        self.datas["train_grouped"] = grouped.values
        logger.info(f"Group By (userID, firstClass) Length: {len(self.datas['train_grouped'])}")

    def data_augmentation(self, choices=[1, 2]):
        """그룹핑을 사용하여 데이터 증강을 합니다.
        1. 테스트 데이터 셋 추가하는 거
        2. UserID + testPaper
        3. UserID + firstClass
        """

        if 2 in choices and 3 in choices:
            raise ValueError("2, 3은 같이 사용 못합니다..!")

        data_aug = {
            1: self._data_augmentation_test_dataset,
            2: self._data_augmentation_user_testpaper,
            3: self._data_augmentation_user_testid,
        }

        for choice in choices:
            data_aug[choice]()

    def feature_engineering(self):
        for key in ["train", "test"]:
            assert self.datas[key] is not None, f"you must loads {key} dataset"
            is_train = True if key == "train" else False
            df = self.fe_pipeline.transform(self.datas[key], is_train=is_train)
            self.datas[key] = df[self.columns]

    def preprocessing(self, pre_encoders):
        """vector 변환
        label   : LabelEncoder
        min_max : MinMaxScaler
        std     : StandardScaler
        """
        assert (
            len(set(pre_encoders.keys()).intersection(set(["label", "min_max", "std"]))) == 3
        ), "pre_encoders의 key값을 ('label', 'min_max', 'std') 다 입력해주세요."

        encoders = {}

        for k in pre_encoders["label"]:
            encoders[k] = LabelEncoder()
            labels = self.datas["train"][k].unique().tolist() + ["unknown"]

            # Train Fit Transform
            self.datas["train"][k] = encoders[k].fit_transform(labels)

            # Valid Transform
            self.datas["valid"][k] = self.datas["valid"][k].apply(lambda x: x if x in labels else "unknown")
            self.datas["valid"][k] = encoders[k].transform(self.datas["valid"][k])

            # Test Transform
            self.datas["test"][k] = self.datas["test"][k].apply(lambda x: x if x in labels else "unknown")
            self.datas["test"][k] = encoders[k].transform(self.datas["test"][k])

        if pre_encoders["min_max"]:
            mm_cols = pre_encoders["min_max"]
            mm_encoder = MinMaxScaler()

            # Train Fit Transform
            self.datas["train"][mm_cols] = mm_encoder.fit_transform(self.datas["train"][mm_cols])

            # Valid Transform
            self.datas["valid"][mm_cols] = mm_encoder.transform(self.datas["valid"][mm_cols])

            # Test Transform
            self.datas["test"][mm_cols] = mm_encoder.transform(self.datas["test"][mm_cols])

        if pre_encoders["std"]:
            std_cols = pre_encoders["std"]
            std_encoder = StandardScaler()

            # Train Fit Transform
            self.datas["train"][std_cols] = std_encoder.fit_transform(self.datas["train"][std_cols])

            # Valid Transform
            self.datas["valid"][std_cols] = std_encoder.transform(self.datas["valid"][std_cols])

            # Test Transform
            self.datas["test"][std_cols] = std_encoder.transform(self.datas["test"][std_cols])


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols
