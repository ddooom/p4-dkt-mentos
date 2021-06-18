import torch

from trainer import DKTTrainer
from dkt_dataset import Preprocess, DKTDataset
from models.lstm.model import LSTM
from fe.feature import FEPipeline

from fe.agg import MakeCorrectCount, MakeCorrectPercent, MakeQuestionCount
from fe.seq import SplitAssessmentItemID, MakeFirstClass, MakeSecondClass, ConvertTime


class FeatureTestTrainer(DKTTrainer):
    def _process_batch(self, batch):
        batch["mask"] = batch["mask"].type(torch.FloatTensor)
        batch["answerCode"] = batch["answerCode"].type(torch.FloatTensor)

        batch["interaction"] = batch["answerCode"] + 1
        batch["interaction"] = batch["interaction"].roll(shifts=1, dims=1)
        batch["mask"] = batch["mask"].roll(shifts=1, dims=1)
        batch["mask"][:, 0] = 0
        batch["interaction"] = (batch["interaction"] * batch["mask"]).to(torch.int64)

        for k in self.args.n_linears:  # 수치형
            batch[k] = batch[k].type(torch.FloatTensor)

        for k, v in self.args.n_embeddings.items():  # 범주형
            batch[k] = batch[k].to(torch.int64)

        for k in batch.keys():
            batch[k] = batch[k].to(self.args.device)

        return batch


def _collate_fn(batches):
    """ key값으로 batch 형성 """
    new_batches = {k: [] for k in batches[0].keys()}

    # batch의 값들을 각 column끼리 그룹화
    for k in batches[0].keys():
        for batch in batches:
            pre_padded = torch.zeros(20)
            pre_padded[-len(batch[k]) :] = batch[k]
            new_batches[k].append(pre_padded)

    for k in batches[0].keys():
        new_batches[k] = torch.stack(new_batches[k])

    return new_batches


def get_dkt_loader(args):
    train_dataset, valid_dataset, test_dataset = get_dkt_dataset(args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    return train_loader, valid_loader, test_loader


def get_dkt_dataset(args):
    train_data, valid_data, test_data = get_simple_data(args)

    train_dataset = DKTDataset(train_data, args, args.columns, is_train=True)
    valid_dataset = DKTDataset(valid_data, args, args.columns, is_train=False)
    test_dataset = DKTDataset(test_data, args, args.columns, is_train=False)

    return train_dataset, valid_dataset, test_dataset


def get_simple_data(args, feature=[], column_name=[], pre_enc=dict()):
    fe_list = [
        SplitAssessmentItemID,
        ConvertTime,
        MakeFirstClass,
        MakeSecondClass,
        MakeCorrectCount,
        MakeQuestionCount,
        MakeCorrectPercent,
    ] + feature

    fe_pipeline = FEPipeline(args, fe_list)
    fe_pipeline.debug()

    columns = ["userID", "answerCode", "testPaper", "timeSec", "firstClass", "secondClass", "correctPer"] + column_name
    pre_encoders = {"label": ["testPaper", "firstClass", "secondClass"], "min_max": ["correctPer"], "std": ["timeSec"]}

    for k in pre_enc.keys():
        pre_encoders[k] += pre_enc[k]

    args.columns = columns[1:]

    preprocess = Preprocess(args, fe_pipeline, columns)

    preprocess.feature_engineering()  # feature engineering
    preprocess.split_data()  # train => train, valid
    preprocess.scaling(pre_encoders)  # vector로 바꿔주고 Scaling해주고
    preprocess.data_augmentation(choices=[1, 3])  # [1, 2] 1: test_data추가, 2: user_id 기준 group_by

    train_data = preprocess.get_data("train_grouped")
    valid_data = preprocess.get_data("valid_grouped")
    test_data = preprocess.get_data("test_grouped")

    return train_data, valid_data, test_data


def train_base_lstm(args, train_data, valid_data, test_data):
    """ Base Feature만 사용하여 학습 """

    trainer = FeatureTestTrainer(args, LSTM)
    auc, acc = trainer.run_cv(train_data, valid_data, test_data, folds=3, seeds=[0, 1, 2])

    print("folds: 3, seeds: [0, 1, 2]")
    print(f"auc: {auc}, acc: {acc}")
    print(f"logging path : {trainer.prefix_save_path}")
