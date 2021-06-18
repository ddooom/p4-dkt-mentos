import os
import json
import random

import torch
import easydict
import numpy as np
import torch.nn as nn
from adamp import AdamP
from madgrad import MADGRAD
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.criterion import LabelSmoothingLoss, F1Loss, FocalLoss


def save_json_file(obj, filedir, filename):
    filepath = os.path.join(filedir, filename)

    with open(filepath, "w") as writer:
        writer.write(json.dumps(obj, indent=4, ensure_ascii=False) + "\n")


def get_optimizer(model, args):
    #  optimizers_fn = {
    #      "sgd": optim.SGD,
    #      "adam": optim.Adam,
    #      "adamW": optim.AdamW,
    #      "adamP": AdamP,  # 기본값을 사용하는 것이 좋다고 한다.
    #      "madgrad": MADGRAD,  # lr 0.005, cliping 도움이 됐다.
    #  }

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 모든 parameter들의 grad값을 0으로 초기화
    #  optimizer = optimizers_fn[args.optimizer](model.parameters(), **args.optimizer_hp)
    #  optimizer.zero_grad()

    return optimizer


def get_scheduler(optimizer, args):
    #  schedulers_fn = {
    #      "cyclic_lr": optim.lr_scheduler.CyclicLR,
    #      "cosine_lr": optim.lr_scheduler.CosineAnnealingLR,
    #      "cosine_warm_lr": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    #      "sgdr": CosineAnnealingWarmupRestarts,
    #      "step_lr": optim.lr_scheduler.StepLR,
    #      "linear_warmup": get_linear_schedule_with_warmup,
    #  }
    #
    #  scheduler = schedulers_fn[args.scheduler](optimizer)(**args.scheduler_hp)
    #  return scheduler

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="max", verbose=True)
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
        )
    return scheduler


#  def get_criterion(args):
#      loss_fns = {
#          "bce_loss": nn.BCELoss,
#          "cross_entropy": nn.CrossEntropyLoss,
#          "f1_loss": F1Loss,
#          "focal_loss": FocalLoss,
#          "smoothing": LabelSmoothingLoss,
#      }
#
#      loss_fn = loss_fns[args.loss](**args.loss_hp)
#      return loss_fn


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)


def get_args():
    config = {}

    # 설정
    config["seed"] = 42
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["use_dynamic"] = False

    # 데이터
    config["max_seq_len"] = 20
    config["num_workers"] = 1
    config["data_dir"] = "/opt/ml/input/data/train_dataset"

    # 모델
    config["hidden_dim"] = 64
    config["n_layers"] = 2
    config["dropout"] = 0.2

    # 훈련
    config["n_epochs"] = 20
    config["batch_size"] = 64
    config["lr"] = 0.0001
    config["clip_grad"] = 10
    config["log_steps"] = 50
    config["patience"] = 5

    # 중요
    config["loss"] = "bce_loss"
    config["loss_hp"] = {"reduction": "none"}
    config["optimizer"] = "adam"
    config["optimizer_hp"] = {"lr": 0.0001, "weight_decay": 0.01}
    config["scheduler"] = "plateau"
    config["scheduler_hp"] = {"patience": 10, "factor": 0.5, "mode": "max", "verbose": True}

    args = easydict.EasyDict(config)
    return args


def get_root_dir(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    return root_dir


def set_seeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
