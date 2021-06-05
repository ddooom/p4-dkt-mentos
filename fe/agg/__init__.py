import os.path as p

import numpy as np
import pandas as pd

from fe.feature import FEBase


class AggFeBase(FEBase):
    fe_type: str = "agg"
    agg_column: list = ["userID"]

    @classmethod
    def transform(cls, df, is_train):
        """ 사용자가 푼 문항수를 만듭니다. """
        save_path = cls.get_save_path(is_train)

        if p.exists(save_path):
            right_df = cls.load_feature_df(save_path)
        else:
            right_df = cls._transform(df)
            cls.save_feature_df(right_df, save_path)

        df = df.merge(right_df, how="inner", on=cls.agg_column)
        return df


class MakeQuestionCount(AggFeBase):
    name = "make_question_count"
    description = {"quesCnt": "사용자가 푼 문항수를 나타냅니다."}

    @classmethod
    def _transform(cls, df):
        """ 사용자가 푼 문항수를 만듭니다. """
        right_df = pd.DataFrame(df.groupby("userID").size(), columns=["quesCnt"]).reset_index()
        return right_df


class MakeCorrectCount(AggFeBase):
    name = "make_correct_count"
    description = {"correctCnt": "사용자가 맞춘 문항수를 나타냅니다."}

    @classmethod
    def _transform(cls, df):
        """ 사용자가 맞춘 문항수를 만듭니다. """
        grouped_df = df.groupby("userID").sum()
        right_df = pd.DataFrame(
            {"userID": list(grouped_df.index), "correctCnt": list(grouped_df.answerCode)}
        )
        return right_df


class MakeCorrectPercent(AggFeBase):
    name = "make_correct_percent"
    pre_fe = {"make_question_count", "make_correct_count"}
    description = {"correctPer": "사용자가 푼 전체 문항에 대한 정답률입니다."}

    @classmethod
    def _transform(cls, df):
        """ 사용자의 정답률을 만듭니다. """
        grouped_df = df.groupby("userID").mean()
        right_df = pd.DataFrame(
            {"userID": list(grouped_df.index), "correctPer": list(grouped_df.correctCnt / grouped_df.quesCnt)}
        )
        return right_df


class MakeTopNCorrectPercent(AggFeBase):
    name = "make_topn_correct_percent"
    description = {
        "top10CorrectPer": "사용자가 최근 푼 TOP-10개에 대한 정답률입니다.",
        "top30CorrectPer": "사용자가 최근 푼 TOP-30개에 대한 정답률입니다.",
        "top50CorrectPer": "사용자가 최근 푼 TOP-50개에 대한 정답률입니다.",
        "top100CorrectPer": "사용자가 최근 푼 TOP-100개에 대한 정답률입니다.",
    }

    @classmethod
    def _transform(cls, df):
        """ 사용자가 최근 푼 TOP-N개에 대한 정답률을 만듭니다. """
        grouped_df = df.groupby("userID")
        right_df = {
            "userID": list(grouped_df.userID.indices.keys()),
            "top10CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-10:]))),
            "top30CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-30:]))),
            "top50CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-50:]))),
            "top100CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-100:]))),
        }

        right_df = pd.DataFrame(right_df)
        return right_df
