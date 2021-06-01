import os.path as p

import pandas as pd
from fe.feature import FEBase


class SeqFeBase(FEBase):
    fe_type = "seq"

    @classmethod
    def transform(cls, df, is_train):
        save_path = cls.get_save_path(is_train)

        if p.exists(save_path):
            right_df = cls.load_feature_df(save_path)
        else:
            right_df = cls._transform(df)
            cls.save_feature_df(right_df, save_path)

        df = pd.concat([df, right_df], axis=1)
        return df


class SplitAssessmentItemID(SeqFeBase):
    name = "split_assessmentitem_id"
    description = {"test_paper": "시험지 번호입니다.", "test_paper_cnt": "시험지의 문항 번호입니다."}

    @classmethod
    def _transform(cls, df):
        """ 시험지 번호,  시험지 내 문항의 번호를 추가합니다. """
        new_df = pd.DataFrame()
        new_df["test_paper"] = df["assessmentItemID"].apply(lambda x: x[1:7])
        new_df["test_paper_cnt"] = df["assessmentItemID"].apply(lambda x: x[7:10])
        return new_df


class MakeFirstClass(SeqFeBase):
    name = "make_first_class"
    description = {"firstClass": "대분류에 해당합니다."}

    @classmethod
    def _transform(cls, df):
        """ 대분류 Feature를 만듭니다. """
        new_df = pd.DataFrame()
        new_df["firstClass"] = df["testId"].apply(lambda x: x[2:3])
        new_df["firstClass"] = new_df["firstClass"].astype(str)
        return new_df


class MakeSecondClass(SeqFeBase):
    name = "make_second_class"
    description = {"secondClass": "중분류에 해당합니다."}

    @classmethod
    def _transform(cls, df):
        """ 중분류 Feature를 만듭니다. """
        new_df = pd.DataFrame()
        new_df["secondClass"] = df["KnowledgeTag"]
        new_df["secondClass"] = new_df["secondClass"].astype(str)
        return new_df


class MakeYMD(SeqFeBase):
    name = "make_year_month_day"
    description = {"YMD": "사용자가 문제를 푼 시점의 '년,월,일'입니다."}

    @classmethod
    def _transform(cls, df):
        new_df = pd.DataFrame()
        new_df["YMD"] = df["Timestamp"].apply(lambda x: x.split(" ")[0])
        return new_df
