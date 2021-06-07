import os.path as p

import numpy as np
import pandas as pd

from fe.feature import FEBase


class AggFeBase(FEBase):
    fe_type = "agg"
    agg_column = ['userID']

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

    
class MakeTagAnswerData(AggFeBase):
    name = "make_tag_answer_data"
    description = {
        "TagCorrectPer": "KnowledgeTag 별 정답률입니다.",
        "TagCorrectSum": "KnowledgeTag 별 정답 문항 수의 합입니다."}
    agg_column = ["KnowledgeTag"]
    
    @classmethod
    def _transform(cls, df):
        grouped_df = df.groupby(cls.agg_column)
        right_df = {
            "KnowledgeTag": list(grouped_df.indices.keys()),
            "TagCorrectPer": list(grouped_df["answerCode"].mean()),
            "TagCorrectSum": list(grouped_df["answerCode"].sum())
        }
        
        right_df = pd.DataFrame(right_df)
        return right_df
    
    
class MakeUserTagAnswerData(AggFeBase):
    name = "make_user_tag_answer_data"
    description = {
        "UserTagCorrectPer": "userID, KnowledgeTag 별 정답률입니다.",
        "UserTagCorrectSum": "userID, KnowledgeTag 별 정답 문항 수의 합입니다."}
    agg_column = ["userID", "KnowledgeTag"]
    
    @classmethod
    def _transform(cls, df):
        grouped_df = df.groupby(cls.agg_column)
        id1,id2 = zip(*grouped_df.indices.keys())
        right_df = {
            "userID": list(id1),
            "KnowledgeTag": list(id2),
            "UserTagCorrectPer": list(grouped_df["answerCode"].mean()),
            "UserTagCorrectSum": list(grouped_df["answerCode"].sum())
        }
        
        right_df = pd.DataFrame(right_df)
        return right_df

    
class MakeTestAnswerData(AggFeBase):
    name = "make_test_answer_data"
    description = {
        "TestCorrectPer": "testId 별 정답률입니다.",
        "TestCorrectSum": "testId 별 정답 문항 수의 합입니다."}
    agg_column = ["testId"]
    
    @classmethod
    def _transform(cls, df):
        grouped_df = df.groupby(cls.agg_column)
        right_df = {
            "testId": list(grouped_df.indices.keys()),
            "TestCorrectPer": list(grouped_df["answerCode"].mean()),
            "TestCorrectSum": list(grouped_df["answerCode"].sum())
        }
        
        right_df = pd.DataFrame(right_df)
        return right_df
    
    
class MakeUserTestAnswerData(AggFeBase):
    name = "make_user_test_answer_data"
    description = {
        "UserTestCorrectPer": "userID, testId 별 정답률입니다.",
        "UserTestCorrectSum": "userID, testId 별 정답 문항 수의 합입니다."}
    agg_column = ["userID", "testId"]
    
    @classmethod
    def _transform(cls, df):
        grouped_df = df.groupby(cls.agg_column)
        id1,id2 = zip(*grouped_df.indices.keys())
        right_df = {
            "userID": list(id1),
            "testId": list(id2),
            "UserTestCorrectPer": list(grouped_df["answerCode"].mean()),
            "UserTestCorrectSum": list(grouped_df["answerCode"].sum())
        }
        
        right_df = pd.DataFrame(right_df)
        return right_df
    
    
class MakeAssessmentAnswerData(AggFeBase):
    name = "make_assessment_answer_data"
    description = {
        "AssessmentCorrectPer": "assessmentItemID 별 정답률입니다.",
        "AssessmentCorrectSum": "assessmentItemID 별 정답 문항 수의 합입니다."}
    agg_column = ["assessmentItemID"]
    
    @classmethod
    def _transform(cls, df):
        grouped_df = df.groupby(cls.agg_column)
        right_df = {
            "assessmentItemID": list(grouped_df.indices.keys()),
            "AssessmentCorrectPer": list(grouped_df["answerCode"].mean()),
            "AssessmentCorrectSum": list(grouped_df["answerCode"].sum())
        }
        
        right_df = pd.DataFrame(right_df)
        return right_df