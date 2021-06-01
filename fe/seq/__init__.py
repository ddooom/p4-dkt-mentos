from fe.feature import FEBase


class SplitAssessmentItemID(FEBase):
    name = "split_assessmentitem_id"
    fe_type = "seq"
    description = {"test_paper": "시험지 번호입니다.", "test_paper_cnt": "시험지의 문항 번호입니다."}

    @classmethod
    def transform(self, df):
        """ 시험지 번호,  시험지 내 문항의 번호를 추가합니다. """
        new_df = df.copy()
        new_df["test_paper"] = df["assessmentItemID"].apply(lambda x: x[1:7])
        new_df["test_paper_cnt"] = df["assessmentItemID"].apply(lambda x: x[7:10])
        return new_df


class MakeFirstClass(FEBase):
    name = "make_first_class"
    fe_type = "seq"
    description = {"firstClass": "대분류에 해당합니다."}

    @classmethod
    def transform(cls, df):
        """ 대분류 Feature를 만듭니다. """
        new_df = df.copy()
        new_df["firstClass"] = df["testId"].apply(lambda x: x[2:3])
        new_df["firstClass"] = new_df["firstClass"].astype(str)
        return df


class MakeSecondClass(FEBase):
    name = "make_second_class"
    fe_type = "seq"
    description = {"secondClass": "중분류에 해당합니다."}

    @classmethod
    def transform(cls, df):
        """ 중분류 Feature를 만듭니다. """
        new_df = df.copy()
        new_df["secondClass"] = df["KnowledgeTag"]
        new_df["secondClass"] = new_df["secondClass"].astype(str)
        return new_df


class MakeYMD(FEBase):
    name = "make_year_month_day"
    fe_type = "seq"
    description = {"YMD": "사용자가 문제를 푼 시점의 '년,월,일'입니다."}

    @classmethod
    def transform(cls, df):
        new_df = df.copy()
        new_df["YMD"] = df["Timestamp"].apply(lambda x: x.split(" ")[0])
        return new_df
