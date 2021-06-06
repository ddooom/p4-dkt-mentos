import torch
import torch.nn as nn

from trainer import DKTTrainer


class EmbeddingLayer(nn.Module):
    def __init__(self, args, hidden_dim):
        super(EmbeddingLayer, self).__init__()

        self.args = args
        self.device = args.device
        self.hidden_dim = hidden_dim

        labels_dim = self.hidden_dim // (len(self.args.n_embeddings) + 1)
        interaction_dim = self.hidden_dim - (labels_dim * len(self.args.n_embeddings))

        self.embedding_interaction = nn.Embedding(3, interaction_dim)
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(v + 1, labels_dim) for k, v in self.args.n_embeddings.items()}  # plus 1 for padding
        )

    def forward(self, batch):
        embed_interaction = self.embedding_interaction(batch["interaction"])
        embed = torch.cat(
            [embed_interaction] + [self.embeddings[k](batch[k]) for k in self.args.n_embeddings.keys()], 2
        )
        return embed


class LinearLayer(nn.Module):
    def __init__(self, args, hidden_dim):
        super(LinearLayer, self).__init__()

        self.args = args
        self.device = args.device

        self.hidden_dim = hidden_dim
        in_features = len(self.args.n_linears)
        self.fc_layer = nn.Linear(in_features, self.hidden_dim)

    def forward(self, batch):
        cont_v = torch.stack([batch[k] for k in self.args.n_linears]).permute(1, 2, 0)
        output = self.fc_layer(cont_v)
        return output


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.emb_layer = EmbeddingLayer(args, self.hidden_dim // 2)
        self.nli_layer = LinearLayer(args, self.hidden_dim // 2)

        self.comb_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, batch):
        batch_size = batch["interaction"].size(0)

        embed = self.emb_layer(batch)
        nnbed = self.nli_layer(batch)

        embed = torch.cat([embed, nnbed], 2)
        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMTrainer(DKTTrainer):
    def _process_batch(self, batch):
        batch["mask"] = batch["mask"].type(torch.FloatTensor)
        batch["answerCode"] = batch["answerCode"].type(torch.FloatTensor)
        batch["correctPer"] = batch["correctPer"].type(torch.FloatTensor)
        batch["timeSec"] = batch["timeSec"].type(torch.FloatTensor)

        batch["interaction"] = batch["answerCode"] + 1
        batch["interaction"] = batch["interaction"].roll(shifts=1, dims=1)
        batch["mask"] = batch["mask"].roll(shifts=1, dims=1)
        batch["mask"][:, 0] = 0
        batch["interaction"] = (batch["interaction"] * batch["mask"]).to(torch.int64)

        batch["testPaper"] = batch["testPaper"].to(torch.int64)
        batch["firstClass"] = batch["firstClass"].to(torch.int64)
        batch["secondClass"] = batch["secondClass"].to(torch.int64)

        for k in batch.keys():
            batch[k] = batch[k].to(self.args.device)

        return batch


if __name__ == "__main__":
    from fe.agg import MakeCorrectCount, MakeCorrectPercent, MakeQuestionCount, MakeTopNCorrectPercent

    from fe.seq import SplitAssessmentItemID, MakeFirstClass, MakeSecondClass, ConvertTime

    from dkt_dataset import Preprocess
    from utils import get_args, get_root_dir
    from fe.feature import FEPipeline
    import ray
    from ray import tune
    import easydict

    ray.init()

    args = get_args()
    args.root_dir = get_root_dir("./hyper_test")
    args.data_dir = "/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/input/data/train_dataset"

    fe_pipeline = FEPipeline(
        args,
        [
            SplitAssessmentItemID,
            ConvertTime,
            MakeFirstClass,
            MakeSecondClass,
            MakeCorrectCount,
            MakeQuestionCount,
            MakeCorrectPercent,
            MakeTopNCorrectPercent,
        ],
    )

    columns = [
        "userID",
        "answerCode",
        "testPaper",
        "timeSec",
        "firstClass",
        "secondClass",
        "correctPer",
        "top10CorrectPer",
    ]
    pre_encoders = {
        "label": ["testPaper", "firstClass", "secondClass"],
        "min_max": ["top10CorrectPer", "correctPer"],
        "std": ["timeSec"],
    }

    preprocess = Preprocess(args, fe_pipeline, columns)

    preprocess.feature_engineering()
    preprocess.split_data()
    preprocess.preprocessing(pre_encoders)
    preprocess.data_augmentation(choices=[1, 3])

    train_dataset = preprocess.get_data("train_grouped")
    valid_dataset = preprocess.get_data("valid_grouped")
    test_dataset = preprocess.get_data("test_grouped")

    args.columns = columns[1:]
    args.hidden_dim = 512

    trainer = LSTMTrainer(args, LSTM)

    tune_args = {
        "metric": "valid_auc",
        "mode": "max",
        "perturbation_interval": 5,  # 5 epoch마다 수행
        "hyperparam_mutations": {
            "hidden_dim": tune.choice([128, 256, 512, 1024, 2048]),
            "n_layers": tune.randint(1, 10),
            "weight_decay": tune.choice([0.001, 0.002, 0.003, 0.004]),
            "batch_size": tune.choice([32, 64, 128]),
            "lr": tune.randn(mean=0.0001, sd=0.0005),
        },
        "quantile_fraction": 0.2,
        "resample_probability": 0.1,
    }
    results = trainer.hyper(args, tune_args, train_dataset, valid_dataset)
    print(results)
