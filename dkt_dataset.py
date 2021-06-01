import os.path as p

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from logger import get_logger

logger = get_logger("preprocess")


class Preprocess:
    def __init__(self, args, fe_pipeline, columns):
        self.args = args
        self.datas = {"train": None, "valid": None, "test": None}
        self.fe_pipeline = fe_pipeline
        self.columns = columns

        assert args.data_dir
        self.datas["train"] = pd.read_csv(p.join(args.data_dir, "train_data.csv"))
        self.datas["test"] = pd.read_csv(p.join(args.data_dir, "test_data.csv"))

        self.encoders = {}

    def get_data(self, key="train"):
        """ return datas """
        assert key in self.datas.keys(), f"{key} not in {self.datas.keys()}"
        assert self.datas[key] is not None, f"{key} dataset is None"
        return self.datas[key]

    def split_data(self):
        """ 서일님이 해 놓은 거 참고 """
        pass

    def _make_rows_to_sequence(self, r):
        datas = []
        for key in self.columns:
            if key == "userID":
                continue
            datas.append(r[key].values)
        return tuple(datas)

    def _data_augmentation_test_dataset(self):
        """ 1 테스트 데이터셋 추가 """
        test_dataset = self.datas["test"]
        aug_test_dataset = test_dataset[test_dataset["userID"] == test_dataset["userID"].shift(-1)]
        self.datas["train"] = pd.concat([self.datas["train"], aug_test_dataset], axis=0)

    def _data_augmentation_user_testpaper(self):
        """ 2 사이클 증강 """
        grouped = self.datas["train"].groupby(["userID", "testPaper"]).apply(lambda r: self._make_rows_to_sequence(r))
        return grouped.values

    def _data_augmentation_user_testid(self):
        """ 3 대분류 증강 """
        grouped = self.datas["train"].groupby(["userID", "firstClass"]).apply(lambda r: self._make_rows_to_sequence(r))
        return grouped.values

    def data_augmentation(self, choices=[1, 2]):
        # 1. 테스트 데이터 셋 추가하는 거
        # 2. UserID + SlideWindow
        # 3.

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

    def preprocessing(self, df, is_train=True):
        cat_list, num_list = [], []  # category, numeric

        # Preprocessing Category
        for key, v in df.dtype.items():
            if v != "object":
                num_list.append(key)
                continue

            cat_list.append(key)

            if is_train:  # fit: 그렇게 오래 걸리는 작업이 아님.
                self.encoders[key] = LabelEncoder()
                labels = df[key].unique().tolist() + ["unknown"]
                self.encoders[key].fit(labels)
            else:
                labels = self.encoders[key].classes_
                df[key] = df[key].apply(lambda x: x if x in labels else "unknown")

            df[key] = df[key].astype(str)
            df[key] = self.encoders[key].transform(df[key])

        # Preprocessing Sequence
        if is_train:
            self.encoders["seq"] = StandardScaler()
            self.encoders["seq"].fit(df[num_list])
            df[num_list] = self.encoders["seq"].transform(df[num_list])
        else:
            df[num_list] = self.encoders["seq"].transform(df[num_list])

        self.args.cat_features = cat_list
        self.args.num_features = num_list

        return df


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
