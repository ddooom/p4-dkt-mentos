import os.path as p

import numpy as np
import pandas as pd

from fe.feature import FEBase


class MakeQuestionCount(FEBase):
    name = "make_question_count"
    fe_type = "agg"
    description = {"quesCnt": "사용자가 푼 문항수를 나타냅니다."}

    @classmethod
    def transform(cls, df):
        """ 사용자가 푼 문항수를 만듭니다. """
        load_path = cls.get_save_path()
        if p.exists(load_path):
            right_df = cls.load_feature_df()
        else:
            right_df = pd.DataFrame(df.groupby("userID").size(), columns=["quesCnt"]).reset_index()
            cls.save_feature_df(right_df)

        df = df.merge(right_df, how="inner", on="userID")
        return df


class MakeCorrectCount(FEBase):
    name = "make_correct_count"
    fe_type = "agg"
    description = {"correctCnt": "사용자가 맞춘 문항수를 나타냅니다."}

    @classmethod
    def transform(cls, df):
        """ 사용자가 맞춘 문항수를 만듭니다. """
        load_path = cls.get_save_path()
        if p.exists(load_path):
            right_df = cls.load_feature_df()
        else:
            grouped_df = df.groupby("userID").sum()
            right_df = pd.DataFrame(
                {"userID": list(grouped_df.index), "correctCnt": list(grouped_df.answerCode)}
            ).reset_index()
            cls.save_feature_df(right_df)

        df = df.merge(right_df, how="inner", on="userID")
        return df


class MakeCorrectPercent(FEBase):
    name = "make_correct_percent"
    pre_fe = {"make_question_count", "make_correct_count"}
    fe_type = "agg"
    description = {"correctPer": "사용자가 푼 전체 문항에 대한 정답률입니다."}

    @classmethod
    def transform(cls, df):
        """ 사용자의 정답률을 만듭니다. """
        df["correctPer"] = df["correctCnt"] / df["quesCnt"]
        return df


class MakeTopNCorrectPercent(FEBase):
    name = "make_topn_correct_percent"
    fe_type = "agg"
    description = {
        "top10CorrectPer": "사용자가 최근 푼 TOP-10개에 대한 정답률입니다.",
        "top30CorrectPer": "사용자가 최근 푼 TOP-30개에 대한 정답률입니다.",
        "top50CorrectPer": "사용자가 최근 푼 TOP-50개에 대한 정답률입니다.",
        "top100CorrectPer": "사용자가 최근 푼 TOP-100개에 대한 정답률입니다.",
    }

    @classmethod
    def transform(cls, df):
        """ 사용자가 최근 푼 TOP-N개에 대한 정답률을 만듭니다. """
        load_path = cls.get_save_path()
        if p.exists(load_path):
            right_df = cls.load_feature_df()
        else:
            grouped_df = df.groupby("userID")
            right_df = {
                "userID": list(grouped_df.userID.indices.keys()),
                "top10CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-10:]))),
                "top30CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-30:]))),
                "top50CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-50:]))),
                "top100CorrectPer": list(grouped_df.apply(lambda x: np.mean(x["answerCode"][-100:]))),
            }
            cls.save_feature_df(right_df)

        right_df = pd.DataFrame(right_df).reset_index()
        df = df.merge(right_df, how="inner", on="userID")
        return df
